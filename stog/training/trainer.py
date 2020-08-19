import re
import os
import shutil
import time
import datetime
import traceback
import sys
from typing import Dict, Optional, List, Union
import torch
from stog.utils import logging
from stog.training.tensorboard import TensorboardWriter
from stog.utils.environment import device_mapping, peak_memory_mb, gpu_memory_mb, move_to_device
from stog.utils.checks import  ConfigurationError
from stog.utils.tqdm import Tqdm
from stog.utils.time import time_to_str
from stog.modules.optimizer import Optimizer
from stog.utils.exception_hook import ExceptionHook
import shutil


sys.excepthook = ExceptionHook()

logger = logging.init_logger()



class Trainer:
    """
    Adopted from AllenNLP:
        https://github.com/allenai/allennlp/blob/v0.6.1/allennlp/training/trainer.py
    """

    def __init__(
            self,
            model,
            optimizer,
            iterator,
            training_dataset,
            dev_dataset = None,
            dev_iterator = None,
            dev_metric = '-loss',
            device = None,
            patience = None,
            grad_clipping = None,
            shuffle = True,
            num_epochs = 20,
            serialization_dir = None,
            num_serialized_models_to_keep = 20,
            model_save_interval = None,
            summary_interval = 100,
            batch_size = 64,
            n_gpus = 0,
    ):
        """
        Parameters
        ----------
        :param model:
            The model to train.
        :param optimizer:
            Optimizer.
        :param iterator:
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        :param training_dataset:
            A ``Dataset`` to train on. The dataset should have already been indexed.
        :param dev_dataset:
            A ``Dataset`` to validate on. The dataset should have already been indexed.
        :param dev_iterator:
            An iterator to use for the dev set.  If ``None``, then
            use the training `iterator`.
        :param dev_metric:
            Dev metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch.
        :param device:
            Specified device.
        :param patience:
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        :param grad_clipping:
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        :param shuffle:
            Whether to shuffle the instances in the iterator or not.
        :param num_epochs:
            Number of training epochs.
        :param serialization_dir:
            Path to save and load model states, training states, and logs.
        :param num_serialized_models_to_keep:
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        :param model_save_interval:
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        :param summary_interval:
            Number of batches between logging scalars to tensorboard
        :param batch_size:
            Training and dev batch size
        :param n_gpus:
            Number of GPUs
        """
        self._model = model
        self._optimizer = optimizer
        self._iterator = iterator
        self._training_dataset = training_dataset
        self._dev_dataset = dev_dataset
        self._dev_iterator = dev_iterator
        self._dev_metric = dev_metric[1:]
        self._dev_metric_decreases = dev_metric[0] == "-"
        self._device = device
        self._patience = patience
        self._grad_clipping = grad_clipping
        self._shuffle = shuffle
        self._num_epochs = num_epochs
        self._serialization_dir = serialization_dir
        self._num_serialized_models_to_keep = num_serialized_models_to_keep
        self._model_save_interval = model_save_interval
        self._summary_interval = summary_interval
        self._batch_size = batch_size
        self._n_gpus = n_gpus

        self._num_trained_batches = 0
        self._serialized_paths = []

        if serialization_dir is not None:
            train_log = os.path.join(serialization_dir, 'log', 'train')
            dev_log = os.path.join(serialization_dir, 'log', 'dev')
            self._tensorboard = TensorboardWriter(train_log, dev_log)
        else:
            self._tensorboard = TensorboardWriter()


    def _batch_loss(self, batch, for_training: bool) -> torch.Tensor:
        batch = move_to_device(batch, self._device)
        output_dict = self._model(batch, for_training=for_training)

        try:
            if self._n_gpus > 1:
                loss = (output_dict["loss"].sum() + output_dict['edge_loss'].sum() / output_dict['num_nodes'].sum())
            else:
                loss = output_dict["loss"]

            if for_training:
                if self._n_gpus > 1:
                    loss += self._model.module.get_regularization_penalty()
                else:
                    loss += self._model.get_regularization_penalty()
       
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    def _train_epoch(self, epoch):
        training_loss = 0.0
        self._model.train()
        train_generator = self._iterator(
            instances=self._training_dataset,
            shuffle=self._shuffle,
            num_epochs=1
        )
        num_training_batches = self._iterator.get_num_batches(self._training_dataset)
        last_save_time = time.time()
        batches_this_epoch = 0
        train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)

        for batch in train_generator_tqdm:
            batches_this_epoch += 1
            self._num_trained_batches += 1
            self._optimizer.zero_grad()
            loss = self._batch_loss(batch, for_training=True)
            loss.backward()
            training_loss += loss.item()
            self._optimizer.step()
      
        return training_loss
    

    def _validate_dev(self, epoch):
        self._model.eval()

        # TODO: edge loss is wrong when _dev_iterator is used.
        if self._dev_iterator is not None:
            dev_iterator = self._dev_iterator
        else:
            dev_iterator = self._iterator

        dev_generator = dev_iterator(
            instances=self._dev_dataset,
            shuffle=False,
            num_epochs=1
        )

        num_dev_batches = dev_iterator.get_num_batches(self._dev_dataset)
        dev_generator_tqdm = Tqdm.tqdm(dev_generator, total=num_dev_batches)
        batches_this_epoch = 0
        dev_loss = 0
        for batch in dev_generator_tqdm:
            batches_this_epoch += 1
            loss = self._batch_loss(batch, for_training=False) #for_training=True)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                dev_loss += loss.item()
   
        return dev_loss
       
    def train(self):
        logger.info('Start training...')
        self._enable_gradient_clipping()

        training_start_time = time.time()
        epochs_trained_this_time = 0
        metrics = {}
        training_metrics = {}
        dev_metrics = {}
        is_best_so_far = True
        best_epoch_dev_metrics = {}

        for epoch in range(0, self._num_epochs):
            epoch_start_time = time.time()
            train_loss = self._train_epoch(epoch)
            if self._dev_dataset is not None:
                with torch.no_grad():
                    dev_loss = self._validate_dev(epoch)
                
            print("epoch:", epoch, " train_loss: ", train_loss, " dev_loss: ", dev_loss)
            self._model.t5.save_pretrained("t5-small-amrtrained_"+str(epoch))
    
        return metrics

    def _enable_gradient_clipping(self) -> None:
        if self._grad_clipping is not None:
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)


    
    @classmethod
    def from_params(cls, model, train_data, dev_data, train_iterator, dev_iterator, params):
        logger.info('Building optimizer..')

        device = params['device']
        optimizer_type = params['optimizer_type']
        lr = params['learning_rate']
        max_grad_norm = params['max_grad_norm']
        dev_metric = params['dev_metric']
        shuffle = params['shuffle']
        epochs = params['epochs']
        serialization_dir = params['serialization_dir']
        model_save_interval = params['model_save_interval']
        batch_size = params['batch_size']
        n_gpus = 1#torch.cuda.device_count()

        if n_gpus > 1:
            logger.info('Multi-GPU ({}) model is enabled!'.format(n_gpus))
            model = torch.nn.DataParallel(model)

        model.to(device)

        optimizer = Optimizer(optimizer_type, lr, max_grad_norm, device=device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer.set_parameters(parameters)

        trainer = cls(
            model=model,
            optimizer=optimizer,
            iterator=train_iterator,
            training_dataset=train_data,
            dev_dataset=dev_data,
            dev_iterator=dev_iterator,
            dev_metric=dev_metric,
            device=device,
            patience=None,
            grad_clipping=None,
            shuffle=shuffle,
            num_epochs=epochs,
            serialization_dir=serialization_dir,
            num_serialized_models_to_keep=5,
            model_save_interval=model_save_interval,
            summary_interval=100,
            batch_size=batch_size,
            n_gpus=n_gpus
        )

        return trainer


