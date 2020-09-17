import os
import re
import argparse
import yaml

import torch

from stog.utils import logging
from stog.utils.params import Params, remove_pretrained_embedding_params
from stog import models as Models
from stog.data.dataset_builder import dataset_from_params, iterator_from_params
from stog.data.vocabulary import Vocabulary
from stog.training.trainer import Trainer
from stog.utils import environment
from stog.utils.checks import ConfigurationError
from stog.utils.archival import CONFIG_NAME, _DEFAULT_WEIGHTS, archive_model
from stog.commands.evaluate import evaluate
from stog.metrics import dump_metrics

logger = logging.init_logger()

# from transformers import T5Tokenizer

# if os.path.isdir("t5-vocab"):
#     t5_tokenizer =  T5Tokenizer.from_pretrained("t5-vocab")
# else:
#     t5_tokenizer = T5Tokenizer.from_pretrained('t5-small', additional_special_tokens=["amrgraphize:"])


# Huggingface
from transformers import T5Tokenizer


def create_serialization_dir(params: Params) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.
    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    """
    serialization_dir = params['environment']['serialization_dir']
    recover = params['environment']['recover']
    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        if not recover:
            raise ConfigurationError(f"Serialization directory ({serialization_dir}) already exists and is "
                                     f"not empty. Specify --recover to recover training from existing output.")

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError("The serialization directory already exists but doesn't "
                                     "contain a config.json. You probably gave the wrong directory.")
        else:
            loaded_params = Params.from_file(recovered_config_file)

            if params != loaded_params:
                raise ConfigurationError("Training configuration does not match the configuration we're "
                                         "recovering from.")

            # In the recover mode, we don't need to reload the pre-trained embeddings.
            remove_pretrained_embedding_params(params)
    else:
        if recover:
            raise ConfigurationError(f"--recover specified but serialization_dir ({serialization_dir}) "
                                     "does not exist.  There is nothing to recover from.")
        os.makedirs(serialization_dir, exist_ok=True)
        params.to_file(os.path.join(serialization_dir, CONFIG_NAME))


def train_model(params: Params):
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results.
    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """

    # Set up the environment.
    environment_params = params['environment']
    environment.set_seed(environment_params)
    create_serialization_dir(params)
    environment.prepare_global_logging(environment_params)
    environment.check_for_gpu(environment_params)
    if environment_params['gpu']:
        device = torch.device('cuda:{}'.format(environment_params['cuda_device']))
        environment.occupy_gpu(device)
    else:
        device = torch.device('cpu')
    params['trainer']['device'] = device


    # Load transformer tokenizer
    if params['transformer_tokenizer'] is None:
        transformer_tokenizer = T5Tokenizer.from_pretrained('t5-small', additional_special_tokens=['amrgraphize:','COUNTRY_1', 'COUNTRY_2', 'COUNTRY_3', 'COUNTRY_4', 'COUNTRY_5', 'COUNTRY_6', 'COUNTRY_7', 'COUNTRY_8', 'COUNTRY_9', 'COUNTRY_10', 'PERSON_1', 'PERSON_2', 'PERSON_3', 'PERSON_4', 'PERSON_5', 'PERSON_6', 'PERSON_7', 'PERSON_8', 'PERSON_9', 'PERSON_10', 'ORGANIZATION_1', 'ORGANIZATION_2', 'ORGANIZATION_3', 'ORGANIZATION_4', 'ORGANIZATION_5', 'ORGANIZATION_6', 'ORGANIZATION_7', 'ORGANIZATION_8', 'ORGANIZATION_9', 'ORGANIZATION_10', 'LOCATION_1', 'LOCATION_2', 'LOCATION_3', 'LOCATION_4', 'LOCATION_5', 'LOCATION_6', 'LOCATION_7', 'LOCATION_8', 'LOCATION_9', 'LOCATION_10', 'HANDLE_1', 'HANDLE_2', 'HANDLE_3', 'HANDLE_4', 'HANDLE_5', 'HANDLE_6', 'HANDLE_7', 'HANDLE_8', 'HANDLE_9', 'HANDLE_10', 'NUMBER_1', 'NUMBER_2', 'NUMBER_3', 'NUMBER_4', 'NUMBER_5', 'NUMBER_6', 'NUMBER_7', 'NUMBER_8', 'NUMBER_9', 'NUMBER_10', 'MISC_1', 'MISC_2', 'MISC_3', 'MISC_4', 'MISC_5', 'MISC_6', 'MISC_7', 'MISC_8', 'MISC_9', 'MISC_10', 'STATE_OR_PROVINCE_1', 'STATE_OR_PROVINCE_2', 'STATE_OR_PROVINCE_3', 'STATE_OR_PROVINCE_4', 'STATE_OR_PROVINCE_5', 'STATE_OR_PROVINCE_6', 'STATE_OR_PROVINCE_7', 'STATE_OR_PROVINCE_8', 'STATE_OR_PROVINCE_9', 'STATE_OR_PROVINCE_10', 'IDEOLOGY_1', 'IDEOLOGY_2', 'IDEOLOGY_3', 'IDEOLOGY_4', 'IDEOLOGY_5', 'IDEOLOGY_6', 'IDEOLOGY_7', 'IDEOLOGY_8', 'IDEOLOGY_9', 'IDEOLOGY_10', 'ENTITY_1', 'ENTITY_2', 'ENTITY_3', 'ENTITY_4', 'ENTITY_5', 'ENTITY_6', 'ENTITY_7', 'ENTITY_8', 'ENTITY_9', 'ENTITY_10', 'CITY_1', 'CITY_2', 'CITY_3', 'CITY_4', 'CITY_5', 'CITY_6', 'CITY_7', 'CITY_8', 'CITY_9', 'CITY_10', 'NATIONALITY_1', 'NATIONALITY_2', 'NATIONALITY_3', 'NATIONALITY_4', 'NATIONALITY_5', 'NATIONALITY_6', 'NATIONALITY_7', 'NATIONALITY_8', 'NATIONALITY_9', 'NATIONALITY_10', 'TITLE_1', 'TITLE_2', 'TITLE_3', 'TITLE_4', 'TITLE_5', 'TITLE_6', 'TITLE_7', 'TITLE_8', 'TITLE_9', 'TITLE_10', 'CAUSE_OF_DEATH_1', 'CAUSE_OF_DEATH_2', 'CAUSE_OF_DEATH_3', 'CAUSE_OF_DEATH_4', 'CAUSE_OF_DEATH_5', 'CAUSE_OF_DEATH_6', 'CAUSE_OF_DEATH_7', 'CAUSE_OF_DEATH_8', 'CAUSE_OF_DEATH_9', 'CAUSE_OF_DEATH_10', 'DATE_1', 'DATE_2', 'DATE_3', 'DATE_4', 'DATE_5', 'DATE_6', 'DATE_7', 'DATE_8', 'DATE_9', 'DATE_10', 'DURATION_1', 'DURATION_2', 'DURATION_3', 'DURATION_4', 'DURATION_5', 'DURATION_6', 'DURATION_7', 'DURATION_8', 'DURATION_9', 'DURATION_10', 'RELIGION_1', 'RELIGION_2', 'RELIGION_3', 'RELIGION_4', 'RELIGION_5', 'RELIGION_6', 'RELIGION_7', 'RELIGION_8', 'RELIGION_9', 'RELIGION_10', 'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'O_6', 'O_7', 'O_8', 'O_9', 'O_10', 'CRIMINAL_CHARGE_1', 'CRIMINAL_CHARGE_2', 'CRIMINAL_CHARGE_3', 'CRIMINAL_CHARGE_4', 'CRIMINAL_CHARGE_5', 'CRIMINAL_CHARGE_6', 'CRIMINAL_CHARGE_7', 'CRIMINAL_CHARGE_8', 'CRIMINAL_CHARGE_9', 'CRIMINAL_CHARGE_10', 'URL_1', 'URL_2', 'URL_3', 'URL_4', 'URL_5', 'URL_6', 'URL_7', 'URL_8', 'URL_9', 'URL_10', 'ORDINAL_1', 'ORDINAL_2', 'ORDINAL_3', 'ORDINAL_4', 'ORDINAL_5', 'ORDINAL_6', 'ORDINAL_7', 'ORDINAL_8', 'ORDINAL_9', 'ORDINAL_10', 'TIME_1', 'TIME_2', 'TIME_3', 'TIME_4', 'TIME_5', 'TIME_6', 'TIME_7', 'TIME_8', 'TIME_9', 'TIME_10', 'SCORE_ENTITY_1', 'SCORE_ENTITY_2', 'SCORE_ENTITY_3', 'SCORE_ENTITY_4', 'SCORE_ENTITY_5', 'SCORE_ENTITY_6', 'SCORE_ENTITY_7', 'SCORE_ENTITY_8', 'SCORE_ENTITY_9', 'SCORE_ENTITY_10', 'ORDINAL_ENTITY_1', 'ORDINAL_ENTITY_2', 'ORDINAL_ENTITY_3', 'ORDINAL_ENTITY_4', 'ORDINAL_ENTITY_5', 'ORDINAL_ENTITY_6', 'ORDINAL_ENTITY_7', 'ORDINAL_ENTITY_8', 'ORDINAL_ENTITY_9', 'ORDINAL_ENTITY_10', 'DATE_ATTRS_1', 'DATE_ATTRS_2', 'DATE_ATTRS_3', 'DATE_ATTRS_4', 'DATE_ATTRS_5', 'DATE_ATTRS_6', 'DATE_ATTRS_7', 'DATE_ATTRS_8', 'DATE_ATTRS_9', 'DATE_ATTRS_10', '_QUANTITY_100_1', '_QUANTITY_100_2', '_QUANTITY_100_3', '_QUANTITY_100_4', '_QUANTITY_100_5', '_QUANTITY_100_6', '_QUANTITY_100_7', '_QUANTITY_100_8', '_QUANTITY_100_9', '_QUANTITY_100_10', '_QUANTITY_1_1', '_QUANTITY_1_2', '_QUANTITY_1_3', '_QUANTITY_1_4', '_QUANTITY_1_5', '_QUANTITY_1_6', '_QUANTITY_1_7', '_QUANTITY_1_8', '_QUANTITY_1_9', '_QUANTITY_1_10', '_QUANTITY_10_1', '_QUANTITY_10_2', '_QUANTITY_10_3', '_QUANTITY_10_4', '_QUANTITY_10_5', '_QUANTITY_10_6', '_QUANTITY_10_7', '_QUANTITY_10_8', '_QUANTITY_10_9', '_QUANTITY_10_10', '_QUANTITY_1000_1', '_QUANTITY_1000_2', '_QUANTITY_1000_3', '_QUANTITY_1000_4', '_QUANTITY_1000_5', '_QUANTITY_1000_6', '_QUANTITY_1000_7', '_QUANTITY_1000_8', '_QUANTITY_1000_9', '_QUANTITY_1000_10', '_QUANTITY_-1.0_1', '_QUANTITY_-1.0_2', '_QUANTITY_-1.0_3', '_QUANTITY_-1.0_4', '_QUANTITY_-1.0_5', '_QUANTITY_-1.0_6', '_QUANTITY_-1.0_7', '_QUANTITY_-1.0_8', '_QUANTITY_-1.0_9', '_QUANTITY_-1.0_10', '_QUANTITY_-547.0_1', '_QUANTITY_-547.0_2', '_QUANTITY_-547.0_3', '_QUANTITY_-547.0_4', '_QUANTITY_-547.0_5', '_QUANTITY_-547.0_6', '_QUANTITY_-547.0_7', '_QUANTITY_-547.0_8', '_QUANTITY_-547.0_9', '_QUANTITY_-547.0_10', '_QUANTITY_-70.0_1', '_QUANTITY_-70.0_2', '_QUANTITY_-70.0_3', '_QUANTITY_-70.0_4', '_QUANTITY_-70.0_5', '_QUANTITY_-70.0_6', '_QUANTITY_-70.0_7', '_QUANTITY_-70.0_8', '_QUANTITY_-70.0_9', '_QUANTITY_-70.0_10', '_QUANTITY_0.25_1', '_QUANTITY_0.25_2', '_QUANTITY_0.25_3', '_QUANTITY_0.25_4', '_QUANTITY_0.25_5', '_QUANTITY_0.25_6', '_QUANTITY_0.25_7', '_QUANTITY_0.25_8', '_QUANTITY_0.25_9', '_QUANTITY_0.25_10'])
    else:
        transformer_tokenizer = T5Tokenizer.from_pretrained(params['transformer_tokenizer'])

    # Load data.
    data_params = params['data']
    dataset = dataset_from_params(data_params, transformer_tokenizer)
    train_data = dataset['train']
    dev_data = dataset.get('dev')
    test_data = dataset.get('test')

    # Vocabulary and iterator are created here.
    vocab_params = params.get('vocab', {})
    vocab = Vocabulary.from_instances(instances=train_data, **vocab_params)
    # Initializing the model can have side effect of expanding the vocabulary
    vocab.save_to_files(os.path.join(environment_params['serialization_dir'], "vocabulary"))

    train_iterator, dev_iterater, test_iterater = iterator_from_params(vocab, data_params['iterator'])

    # Build the model.
    model_params = params['model']
    model = getattr(Models, model_params['model_type']).from_params(vocab, model_params, transformer_tokenizer)
    logger.info(model)

    # Train
    trainer_params = params['trainer']
    no_grad_regexes = trainer_params['no_grad']
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = \
        environment.get_frozen_and_tunable_parameter_names(model)
    logger.info("Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info("Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)

    trainer = Trainer.from_params(model, train_data, dev_data, train_iterator, dev_iterater, trainer_params)

    serialization_dir = trainer_params['serialization_dir']
    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logger.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir)
        raise

    # Now tar up results
    #archive_model(serialization_dir)

    logger.info("Training is over...")

    # logger.info("Loading the best epoch weights.")
    # best_model_state_path = os.path.join(serialization_dir, 'best.th')
    # best_model_state = torch.load(best_model_state_path)
    # best_model = model
    # if not isinstance(best_model, torch.nn.DataParallel):
    #     best_model_state = {re.sub(r'^module\.', '', k):v for k, v in best_model_state.items()}
    # best_model.load_state_dict(best_model_state)

    # return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('params', help='Parameters YAML file.')
    args = parser.parse_args()

    params = Params.from_file(args.params)
    logger.info(params)

    train_model(params)
