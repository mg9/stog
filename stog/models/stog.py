import torch

from stog.modules.seq2seq_encoders import Seq2SeqBertEncoder

from stog.models.model import Model
from stog.utils.logging import init_logger
from stog.modules.token_embedders.embedding import Embedding
from stog.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from stog.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from stog.modules.stacked_bilstm import StackedBidirectionalLstm
from stog.modules.stacked_lstm import StackedLstm
from stog.modules.decoders.rnn_decoder import InputFeedRNNDecoder
from stog.modules.attention_layers.global_attention import GlobalAttention
from stog.modules.attention import DotProductAttention
from stog.modules.attention import MLPAttention
from stog.modules.attention import BiaffineAttention
from stog.modules.input_variational_dropout import InputVariationalDropout
from stog.modules.decoders.generator import Generator
from stog.modules.decoders.pointer_generator import PointerGenerator
from stog.modules.decoders.deep_biaffine_graph_decoder import DeepBiaffineGraphDecoder
from stog.utils.nn import get_text_field_mask
from stog.utils.string import START_SYMBOL, END_SYMBOL, find_similar_token, is_abstract_token
from stog.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from stog.data.tokenizers.character_tokenizer import CharacterTokenizer
# The following imports are added for mimick testing.
from stog.data.dataset_builder import load_dataset_reader
from stog.predictors.predictor import Predictor
from stog.commands.predict import _PredictManager
import subprocess
import math, os, re

from transformers import T5Tokenizer, T5Model, T5Config
from transformers import EncoderDecoderModel

logger = init_logger()


class STOG(Model):

    def __init__(self,
                 vocab,
                 punctuation_ids,
                 use_char_cnn,
                 max_decode_length,
                 # Transformers
                 transformers,
                 transformer_tokenizer,
                 amrnodes_tot5_tokens,
                 # Generator
                 generator,
                 # Graph decoder
                 graph_decoder,
                 test_config
                 ):
        super(STOG, self).__init__()

        self.vocab = vocab
        self.punctuation_ids = punctuation_ids
        self.use_char_cnn = use_char_cnn
        self.max_decode_length = max_decode_length

        self.transformers = transformers
        self.transformer_tokenizer = transformer_tokenizer
        self.amrnodes_tot5_tokens = amrnodes_tot5_tokens
        self.generator = generator
        self.graph_decoder = graph_decoder

<<<<<<< HEAD
        self.beam_size = 1
=======
        self.beam_size = 0
>>>>>>> tmp
        self.test_config = test_config

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def set_decoder_token_indexers(self, token_indexers):
        self.decoder_token_indexers = token_indexers
        self.character_tokenizer = CharacterTokenizer()

    def get_metrics(self, reset: bool = False, mimick_test: bool = False):
        metrics = dict()
        if mimick_test and self.test_config:
            metrics = self.mimick_test()
        generator_metrics = self.generator.metrics.get_metric(reset)
        graph_decoder_metrics = self.graph_decoder.metrics.get_metric(reset)
        metrics.update(generator_metrics)
        metrics.update(graph_decoder_metrics)
        if 'F1' not in metrics:
            metrics['F1'] = metrics['all_acc']
        return metrics

    def mimick_test(self):
        word_splitter = None
        # if self.use_bert:
        #     word_splitter = self.test_config.get('word_splitter', None)
        dataset_reader = load_dataset_reader('AMR', word_splitter=word_splitter, transformer_tokenizer= self.transformer_tokenizer, amrnodes_tot5_tokens= self.amrnodes_tot5_tokens)
        dataset_reader.set_evaluation()
        predictor = Predictor.by_name('STOG')(self, dataset_reader)
        manager = _PredictManager(
            predictor,
            self.test_config['data'],
            self.test_config['prediction'],
            self.test_config['batch_size'],
            False,
            True,
            0 # 1 beam size
        )
        try:
            logger.info('Mimicking test...')
            manager.run()
        except Exception as e:
            logger.info('Exception threw out when running the manager.')
            logger.error(e, exc_info=True)
            return {}
        try:
            logger.info('Computing the Smatch score...')
            result = subprocess.check_output([
                self.test_config['eval_script'],
                self.test_config['smatch_dir'],
                self.test_config['data'],
                self.test_config['prediction']
            ]).decode().split()
            result = list(map(float, result))
            return dict(PREC=result[0]*100, REC=result[1]*100, F1=result[2]*100)
        except Exception as e:
            logger.info('Exception threw out when computing smatch.')
            logger.error(e, exc_info=True)
            return {}

    def prepare_batch_input(self, batch):

        # [batch, num_tokens]
        encoder_token_inputs =  batch['src_ids']
        encoder_mask = (batch['src_ids'] != 0).long()

        encoder_inputs = dict(
            token=encoder_token_inputs,
            mask=encoder_mask
        )

        # [batch, num_tokens]
        decoder_token_inputs = batch['tgt_ids'][:, :-1] # batch['tgt_tokens']['decoder_tokens']#[:, :-1].contiguous()
        decoder_pos_tags = batch['tgt_pos_tags']#[:, :-1]
        # [batch, num_tokens, num_chars]
        # TODO: The following change can be done in amr.py.
        # Initially, raw_coref_inputs has value like [0, 0, 0, 1, 0]
        # where '0' indicates that the input token has no precedent, and
        # '1' indicates that the input token's first precedent is at position '1'.
        # Here, we change it to [0, 1, 2, 1, 4] which means if the input token
        # has no precedent, then it is referred to itself.
        raw_coref_inputs = batch["tgt_copy_indices"].contiguous() #[:, :-1].contiguous()
        coref_happen_mask = raw_coref_inputs.ne(0)
        decoder_coref_inputs = torch.ones_like(raw_coref_inputs) * torch.arange(
            0, raw_coref_inputs.size(1)).type_as(raw_coref_inputs).unsqueeze(0)
        decoder_coref_inputs.masked_fill_(coref_happen_mask, 0)
        # [batch, num_tokens]
        decoder_coref_inputs = decoder_coref_inputs + raw_coref_inputs

        decoder_inputs = dict(
            token=decoder_token_inputs,
            pos_tag=decoder_pos_tags,
            coref=decoder_coref_inputs
        )

        # [batch, num_tokens]
        vocab_targets = batch['tgt_tokens']['decoder_tokens'][:, 1:].contiguous()
        # [batch, num_tokens]
        coref_targets = batch["tgt_copy_indices"][:, 1:]
        # [batch, num_tokens, num_tokens + coref_na]
        coref_attention_maps = batch['tgt_copy_map'][:, 1:]  # exclude BOS
        # [batch, num_tgt_tokens, num_src_tokens + unk]
        copy_targets = batch["src_copy_indices"][:, 1:]

<<<<<<< HEAD
=======
        # print("normal tokens.shape: ", batch['tgt_tokens']['decoder_tokens'].shape)
        # print("normal tokens: ", batch['tgt_tokens']['decoder_tokens'])
        # print("decoder_token_inputs.shape:", decoder_token_inputs.shape)
        # print("decoder_token_inputs:", decoder_token_inputs)
        # print("vocab_targets:", vocab_targets)
        # print("coref_targets:", coref_targets)
        # print("copy_targets:",  copy_targets)


>>>>>>> tmp
        # [batch, num_src_tokens + unk, src_dynamic_vocab_size]
        # Exclude the last pad.
        copy_attention_maps = batch['src_copy_map'][:, 1:-1] #1:-1]

        generator_inputs = dict(
            vocab_targets=vocab_targets,
            coref_targets=coref_targets,
            coref_attention_maps=coref_attention_maps,
            copy_targets=copy_targets,
            copy_attention_maps=copy_attention_maps
        )
       
        # print("batch['head_indices'].shape: ", batch['head_indices'].shape)
        # print("batch['head_indices']: ", batch['head_indices'])
        # print("batch['head_tags'].shape: ", batch['head_tags'].shape)
        # print("batch['head_tags']: ", batch['head_tags'])
       


        # Remove the last two pads so that they have the same size of other inputs?
        edge_heads = batch['head_indices'][:,:-2]#[:, :-2]
        edge_labels = batch['head_tags'][:,:-2]#[:, :-2]
        # TODO: The following computation can be done in amr.py.
        # Get the parser mask.
        parser_token_inputs = torch.zeros_like(decoder_token_inputs)
        parser_token_inputs.copy_(decoder_token_inputs)
        parser_token_inputs[
            parser_token_inputs == 1 #self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')
        ] = 0
        parser_mask = (parser_token_inputs != 0).float()
        parser_mask[:,0] = 1

        # print("parser_token_inputs.shape: ", parser_token_inputs.shape)
        # print("parser_token_inputs: ", parser_token_inputs)
        # print("parser_mask.shape: ", parser_mask.shape)
        # print("parser_mask: ", parser_mask)
        
        # print("edge_heads.shape: ", edge_heads.shape)
        # print("edge_heads: ", edge_heads)
       
        # print("edge_labels.shape: ", edge_labels.shape)
        # print("edge_labels: ", edge_labels)

        parser_inputs = dict(
            edge_heads=edge_heads,
            edge_labels=edge_labels,
            corefs=decoder_coref_inputs,
            mask=parser_mask
        )
        return encoder_inputs, decoder_inputs, generator_inputs, parser_inputs

    def forward(self, batch, for_training=False):
        encoder_inputs, decoder_inputs, generator_inputs, parser_inputs = self.prepare_batch_input(batch)

        if for_training:
            self.transformers.train()

            enc_dec_outputs = self.encode_decode(
                                encoder_inputs['token'],
                                decoder_inputs['token'],
                                encoder_inputs['mask'],
                                parser_inputs['mask']
                            )

            generator_output = self.generator(
                enc_dec_outputs['decoder_hiddens'],
                enc_dec_outputs['copy_attentions'],
                generator_inputs['copy_attention_maps'],
                enc_dec_outputs['coref_attentions'],
                generator_inputs['coref_attention_maps']
            )

            generator_loss_output = self.generator.compute_loss(
                generator_output['probs'],
                generator_output['predictions'],
                generator_inputs['vocab_targets'],
                generator_inputs['copy_targets'],
                generator_output['source_dynamic_vocab_size'],
                generator_inputs['coref_targets'],
                generator_output['target_dynamic_vocab_size'],
                None,
                enc_dec_outputs['copy_attentions']
            )

            graph_decoder_outputs = self.graph_decode(
                enc_dec_outputs['decoder_hiddens'],
                parser_inputs['edge_heads'],
                parser_inputs['edge_labels'],
                parser_inputs['corefs'],
                parser_inputs['mask']
            )

            return dict(
                loss=generator_loss_output['loss'] + graph_decoder_outputs['loss'],
                token_loss=generator_loss_output['total_loss'],
                edge_loss=graph_decoder_outputs['total_loss'],
                num_tokens=generator_loss_output['num_tokens'],
                num_nodes=graph_decoder_outputs['num_nodes']
            )

        else:
            self.transformers.eval()
            source_copy_invalids = batch.get('source_copy_invalid_ids', None)

            # Disable copying the tokens not exists in decoder_token_ids
            for bi,b in enumerate(batch['src_copy_vocab']):
                for i,token in b.idx_to_token.items():
                    decoder_vocab_id = self.vocab.get_token_index(token, 'decoder_token_ids')
                    if decoder_vocab_id == 1:
                        source_copy_invalids[bi].add(i)

            ## Disable copying the tokens not exists in decoder_token_ids
            source_copy_invalids = batch.get('source_copy_invalid_ids', None)
            for bi,b in enumerate(batch['src_copy_vocab']):
                for i,token in b.idx_to_token.items():
                    decoder_vocab_id = self.vocab.get_token_index(token, 'decoder_token_ids')
                    if decoder_vocab_id == 1:
                        source_copy_invalids[bi].add(i)

            invalid_indexes = dict(
                source_copy=source_copy_invalids,
                vocab= [set(self.punctuation_ids) for _ in range(len(batch['tag_lut']))]
            )

            return dict(
                src_tokens = encoder_inputs['token'],
                encoder_mask= encoder_inputs['mask'],
                copy_attention_maps=generator_inputs['copy_attention_maps'],
                copy_vocabs=batch['src_copy_vocab'],
                tag_luts=batch['tag_lut'],
                invalid_indexes=invalid_indexes
            )

    def encode_decode(self, src_tokens, tgt_tokens, src_mask, tgt_mask): 

<<<<<<< HEAD
=======
        # print("src_tokens.shape: ", src_tokens.shape)
        # print("src_tokens: ", src_tokens)
        # print("src_mask.shape: ", src_mask.shape)
        # print("src_mask: ", src_mask)
        # print("tgt_tokens.shape: ", tgt_tokens.shape)
        # print("tgt_tokens: ", tgt_tokens)
        # print("tgt_mask.shape: ", tgt_mask.shape)
        # print("tgt_mask: ", tgt_mask)

>>>>>>> tmp
        outputs = self.transformers(input_ids=src_tokens, decoder_input_ids=tgt_tokens, attention_mask=src_mask, decoder_attention_mask=tgt_mask, 
                                    output_attentions=True, output_hidden_states=True, return_dict=True)
        encoder_hiddens = outputs.encoder_last_hidden_state
        decoder_hiddens = outputs.last_hidden_state

        coref_attentions = outputs.decoder_attentions[10] 
        coref_attentions = torch.sum(coref_attentions, dim=1)      # B,Ty,Ty
        
        copy_attentions = outputs.decoder_attentions[11] 
        copy_attentions = torch.sum(copy_attentions, dim=1)        # B,Ty,Tx

        return dict(
            decoder_hiddens= decoder_hiddens,                              # torch.Size([B, Ty, H])
            coref_attentions=coref_attentions,                             # torch.Size([B, Ty, Ty])
            copy_attentions=copy_attentions                                # torch.Size([B, Ty, Tx])
        )

    def graph_decode(self, memory_bank, edge_heads, edge_labels, corefs, mask):
        # Exclude the BOS symbol.
        memory_bank = memory_bank[:, 1:]
        corefs = corefs[:, 1:]
        mask = mask[:, 1:]
        return self.graph_decoder(memory_bank, edge_heads, edge_labels, corefs, mask)

    def decode(self, input_dict):

        src_tokens = input_dict['src_tokens']
        mask = input_dict['encoder_mask']
        copy_attention_maps = input_dict['copy_attention_maps']
        copy_vocabs = input_dict['copy_vocabs']
        tag_luts = input_dict['tag_luts']
        invalid_indexes = input_dict['invalid_indexes']
       
        if self.beam_size == 0:
            generator_outputs = self.decode_with_pointer_generator(src_tokens, mask, copy_attention_maps, copy_vocabs,  tag_luts, invalid_indexes)
        else:
            generator_outputs = self.beam_search_with_pointer_generator(src_tokens, mask, copy_attention_maps, copy_vocabs,  tag_luts, invalid_indexes)

        parser_outputs = self.decode_with_graph_parser(
            generator_outputs['decoder_memory_bank'],
            generator_outputs['coref_indexes'],
            generator_outputs['decoder_mask']
        )

        return dict(
            nodes= generator_outputs['predictions'],
            heads=parser_outputs['edge_heads'],
            head_labels=parser_outputs['edge_labels'],
            corefs=generator_outputs['coref_indexes']
        )

    def beam_search_with_pointer_generator(self, encoder_inputs, mask, copy_attention_maps, copy_vocabs, tag_luts, invalid_indices):
        ## TODO: Refactor for t5 vocab in decoder
        batch_size = encoder_inputs.size(0)
        beam_size = self.beam_size
       
        # print("encoder_inputs.shape: ", encoder_inputs.shape)
        # print("encoder_inputs: ", encoder_inputs)
     
        # new_order is used to replicate tensors for different beam
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, beam_size).view(-1).type_as(mask)

        # special token indices
        bos_token = self.vocab.get_token_index(START_SYMBOL, "decoder_token_ids")
        eos_token = self.vocab.get_token_index(END_SYMBOL, "decoder_token_ids")
        pad_token = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, "decoder_token_ids")

        bucket = [[] for i in range(batch_size)]
        bucket_max_score = [-1e8 for i in range(batch_size)]


        def flatten(tensor):
            sizes = list(tensor.size())
            assert len(sizes) >= 2
            assert sizes[0] == batch_size and sizes[1] == beam_size

            if len(sizes) == 2:
                new_sizes = [sizes[0] * sizes[1], 1]
            else:
                new_sizes = [sizes[0] * sizes[1]] + sizes[2:]

            return tensor.contiguous().view(new_sizes)

        def fold(tensor):
            sizes = list(tensor.size())
            new_sizes = [batch_size, beam_size]

            if len(sizes) >= 2:
                new_sizes = [batch_size, beam_size] + sizes[1:]

            return tensor.view(new_sizes)

        def beam_select_2d(input, indices):
            # input [batch_size, beam_size, ......]
            # indices [batch_size, beam_size]
            input_size = list(input.size())
            indices_size = list(indices.size())
            assert len(indices_size) == 2
            assert len(input_size) >= 2
            assert input_size[0] == indices_size[0]
            assert input_size[1] == indices_size[1]

            return input.view(
                [input_size[0] * input_size[1]] + input_size[2:]
            ).index_select(
                0,
                (
                        torch.arange(
                            indices_size[0]
                        ).unsqueeze(1).expand_as(indices).type_as(indices) * indices_size[1] + indices
                ).view(-1)
            ).view(input_size)

        def beam_select_1d(input, indices):
            input_size = list(input.size())
            indices_size = list(indices.size())
            assert len(indices_size) == 2
            assert len(input_size) > 1
            assert input_size[0] == indices_size[0] * indices_size[1]

            return input.index_select(
                0,
                (
                    torch.arange(
                        indices_size[0]
                    ).unsqueeze(1).expand_as(indices).type_as(indices) * indices_size[1] + indices
                ).view(-1)
            ).view(input_size)

        def update_tensor_buff(key, step, beam_indices, tensor, select_input=True):
            if step == 0 and beam_buffer[key] is None:
                beam_buffer[key] = tensor.new_zeros(
                    batch_size,
                    beam_size,
                    self.max_decode_length,
                    tensor.size(-1)
                )

            if select_input:
                beam_buffer[key][:, :, step] = fold(tensor.squeeze(1))
                beam_buffer[key] = beam_select_2d(beam_buffer[key], beam_indices)
            else:
                beam_buffer[key] = beam_select_2d(beam_buffer[key], beam_indices)
                beam_buffer[key][:, :, step] = fold(tensor.squeeze(1))

        # def get_decoder_input(tokens, pos_tags, corefs):
        #     token_embeddings = self.decoder_token_embedding(tokens)
        #     pos_tag_embeddings = self.decoder_pos_embedding(pos_tags)
        #     coref_embeddings = self.decoder_coref_embedding(corefs)

        #     if self.use_char_cnn:
        #         # TODO: get chars from tokens.
        #         # [batch_size, 1, num_chars]
        #         chars = character_tensor_from_token_tensor(
        #             tokens,
        #             self.vocab,
        #             self.character_tokenizer
        #         )

        #         char_cnn_output = self._get_decoder_char_cnn_output(chars)
        #         decoder_inputs = torch.cat(
        #             [token_embeddings, pos_tag_embeddings,
        #              coref_embeddings, char_cnn_output], 2)
        #     else:
        #         decoder_inputs = torch.cat(
        #             [token_embeddings, pos_tag_embeddings, coref_embeddings], 2)

        #     return self.decoder_embedding_dropout(decoder_inputs)

        def repeat_list_item(input_list, n):
            new_list = []
            for item in input_list:
                new_list += [item] * n
            return new_list

        beam_buffer = {}
        beam_buffer["predictions"] = mask.new_full(
            (batch_size, beam_size, self.max_decode_length),
            pad_token
        )

        beam_buffer["coref_indexes"] = copy_attention_maps.new_zeros(
            batch_size,
            beam_size,
            self.max_decode_length
        )

        beam_buffer["decoder_mask"] = copy_attention_maps.new_ones(
            batch_size,
            beam_size,
            self.max_decode_length
        )


        # beam_buffer["decoder_inputs"] = None
        beam_buffer["decoder_memory_bank"] = None
        beam_buffer["scores"] = copy_attention_maps.new_zeros(batch_size, beam_size, 1)
        beam_buffer["scores"][:, 1:] = -float(1e8)

        # inter media variables
        variables = {}

        variables["input_tokens"] = beam_buffer["predictions"].new_full(
            (batch_size * beam_size, 1),
            bos_token
        )

        variables["pos_tags"] = mask.new_full(
            (batch_size * beam_size, 1),
            self.vocab.get_token_index(DEFAULT_OOV_TOKEN, "pos_tags")
        )

        variables["corefs"] = mask.new_zeros(batch_size * beam_size, 1)

        # A sparse indicator matrix mapping each node to its index in the dynamic vocab.
        # Here the maximum size of the dynamic vocab is just max_decode_length.
        variables["coref_attention_maps"] = copy_attention_maps.new_zeros(
            batch_size * beam_size, self.max_decode_length, self.max_decode_length + 1
        )
        # A matrix D where the element D_{ij} is for instance i the real vocab index of
        # the generated node at the decoding step `i'.
        variables["coref_vocab_maps"] = mask.new_zeros(batch_size * beam_size, self.max_decode_length + 1)

        for key in invalid_indices.keys():
            invalid_indices[key] = repeat_list_item(invalid_indices[key], beam_size)

        for step in range(self.max_decode_length):  # one extra step for EOS marker
            # 1. Decoder inputs
            # decoder_inputs : [ batch_size * beam_size, model_dim]
            tokens = variables["input_tokens"]

            tgt_ids =[] 
            tgt_sequences = []
            for b in range(tokens.size(0)):
                sequence = []
                for t in range(tokens.size(1)):
                    token = self.vocab.get_token_from_index(tokens[b,t].item(), "decoder_token_ids")
                    # print("token: ", token)
                    result = re.search("-[0-9]", token)
                    if result is not None:
                        s,e = result.span()
                        token = token[:s]
                    if token in self.amrnodes_tot5_tokens.keys():
                        token = self.amrnodes_tot5_tokens[token].strip()
                    else:
                        token = "<unk>"
                    tokenid = self.transformer_tokenizer.convert_tokens_to_ids(token)
                    if tokenid == 2:
                        tokenid = self.transformer_tokenizer.convert_tokens_to_ids("▁"+token)
                    # print("token -> ", token, " id: ", tokenid)
                    # print("token: ", token, " id: ", tokenid)
                    sequence.append(tokenid) 
                tgt_sequences.append(sequence)
            tgt_ids = torch.reshape(torch.tensor(tgt_sequences), (batch_size* beam_size, step+1)).cuda()

            # print("tgt_ids: ", tgt_ids)

            # get_decoder_input(
            #     variables["input_tokens"],
            #     variables["pos_tags"],
            #     variables["corefs"]
            # )

            # # 2. Decode one stepi.
            # print("\nstep: ", step)
        
            outputs = self.transformers(input_ids=encoder_inputs.index_select(0, new_order), attention_mask=mask.index_select(0, new_order), decoder_input_ids=tgt_ids, output_attentions=True, output_hidden_states=True, return_dict=True)
            decoder_hiddens = outputs.last_hidden_state
            
            _copy_attentions = outputs.decoder_attentions[11] 
            _copy_attentions = torch.sum(_copy_attentions, dim=1)        # B,Ty,Tx

            _coref_attentions = outputs.decoder_attentions[10]  
            _coref_attentions = torch.sum(_coref_attentions, dim=1)      # B,Ty,Ty

            # 3. Run pointer/generator.instance.fields['src_copy_vocab'].metadata
            if step == 0:
                _coref_attention_maps = variables["coref_attention_maps"][:, :step + 1]
            else:
                _coref_attention_maps = variables["coref_attention_maps"][:, :step]
                decoder_hiddens = decoder_hiddens[:,-1:,:]
                _copy_attentions = _copy_attentions[:, -1:,:]
                _coref_attentions = _coref_attentions[:, -1:, :-1]


            generator_output = self.generator(
                decoder_hiddens,
                _copy_attentions,
                copy_attention_maps.index_select(0, new_order),
                _coref_attentions,
                _coref_attention_maps,
                invalid_indices
            )

            word_lprobs = fold(torch.log(1e-8 + generator_output['probs'].squeeze(1)))
            new_all_scores = word_lprobs + beam_buffer["scores"].expand_as(word_lprobs) 
     
            # top beam_size hypos
            # new_hypo_indices : [batch_size, beam_size * 2]
            new_hypo_scores, new_hypo_indices = torch.topk(
                new_all_scores.view(batch_size, -1).contiguous(),
                beam_size, #k
                dim=-1
            )

            # print("new_hypo_scores.shape: ", new_hypo_scores.shape)
            # print("new_hypo_scores: ", new_hypo_scores)


            new_token_indices = torch.fmod(new_hypo_indices, word_lprobs.size(-1))
            # print("new_token_indices.shape: ", new_token_indices.shape)
            # print("new_token_indices: ", new_token_indices)



            eos_token_mask = new_token_indices.eq(eos_token)

            # print("eos_token_mask.shape: ", eos_token_mask.shape)
            # print("eos_token_mask: ", eos_token_mask)

            eos_beam_indices_offset = torch.div(
                new_hypo_indices,
                word_lprobs.size(-1)
            )[:, :beam_size] + new_order.view(batch_size, beam_size) * beam_size


            # print("eos_beam_indices_offset.shape: ", eos_beam_indices_offset.shape)
            # print("eos_beam_indices_offset: ", eos_beam_indices_offset)
            eos_beam_indices_offset = eos_beam_indices_offset.masked_select(eos_token_mask[:, :beam_size])
            # print("eos_beam_indices_offset_after.shape: ", eos_beam_indices_offset.shape)
            # print("eos_beam_indices_offset_after: ", eos_beam_indices_offset)

            if eos_beam_indices_offset.numel() > 0:
                for index in eos_beam_indices_offset.tolist():
                    eos_batch_idx = int(index / beam_size)
                    eos_beam_idx = index % beam_size

                    # print("index: ", index)
                    # print("eos_batch_idx: ", eos_batch_idx)
                    # print("eos_beam_idx: ", eos_beam_idx)

                    hypo_score = float(new_hypo_scores[eos_batch_idx, eos_beam_idx]) / (step + 1)

                    # print("hypo_score: ", hypo_score)

                    if step > 0 and hypo_score > bucket_max_score[eos_batch_idx] and eos_beam_idx == 0:

                        # print("yes! ",)

                        bucket_max_score[eos_batch_idx] = hypo_score
                        bucket[eos_batch_idx] += [
                            {
                                key: tensor[eos_batch_idx, eos_beam_idx].unsqueeze(0) for key, tensor in beam_buffer.items()
                            }
                        ]
                        #bucket[eos_batch_idx][-1]['decoder_inputs'][0, step] = decoder_inputs[index, 0]
                        #bucket[eos_batch_idx][-1]['decoder_rnn_memory_bank'][0, step] = _rnn_outputs[index, 0]
                        #bucket[eos_batch_idx][-1]['decoder_memory_bank'][0, step] = _decoder_outputs[index, 0]
                        #bucket[eos_batch_idx][-1]['decoder_mask'][0, step] = 1

                eos_token_mask = eos_token_mask.type_as(new_hypo_scores)
                active_hypo_scores, active_sort_indices = torch.sort(
                    (1 - eos_token_mask) * new_hypo_scores + eos_token_mask * - float(1e8),
                    descending = True
                )

                active_sort_indices_offset = active_sort_indices \
                    + beam_size * torch.arange(
                        batch_size
                    ).unsqueeze(1).expand_as(active_sort_indices).type_as(active_sort_indices)
                active_hypo_indices = new_hypo_indices.view(batch_size * beam_size)[
                    active_sort_indices_offset.view(batch_size * beam_size)
                ].view(batch_size, -1)

                # print("active_hypo_scores.shape: ", active_hypo_scores.shape)
                # print("active_hypo_scores: ", active_hypo_scores)
                # print("active_hypo_indices.shape: ", active_hypo_indices.shape)
                # print("active_hypo_indices: ", active_hypo_indices)

                new_hypo_scores = active_hypo_scores
                new_hypo_indices = active_hypo_indices
                new_token_indices = torch.fmod(new_hypo_indices, word_lprobs.size(-1))

                # print("new_token_indices.shape: ", new_token_indices.shape)
                # print("new_token_indices: ", new_token_indices)
                

            # find out which beam the new hypo came from and what is the new token
            beam_indices = torch.div(new_hypo_indices, word_lprobs.size(-1))
            # print("beam_indices.shape: ", beam_indices.shape)
            # print("beam_indices: ", beam_indices)

            if step == 0:
                decoder_mask_input = []
            else:
                decoder_mask_input = beam_select_2d(
                    beam_buffer["decoder_mask"],
                    beam_indices
                ).view(batch_size * beam_size, -1)[:, :step].split(1, 1)


            variables["coref_attention_maps"] = beam_select_1d(variables["coref_attention_maps"], beam_indices)
            variables["coref_vocab_maps"] = beam_select_1d(variables["coref_vocab_maps"], beam_indices)

            # print("src dyn vcb size: ",  generator_output['source_dynamic_vocab_size'])
            _input_tokens, _predictions, _pos_tags, _corefs, _mask = self._update_maps_and_get_next_input(
                step,
                flatten(new_token_indices).squeeze(1),
                generator_output['source_dynamic_vocab_size'],
                variables["coref_attention_maps"],
                variables["coref_vocab_maps"],
                repeat_list_item(copy_vocabs, beam_size),
                decoder_mask_input,
                repeat_list_item(tag_luts, beam_size),
                invalid_indices
            )

            beam_buffer["scores"] = new_hypo_scores.unsqueeze(2)

            update_tensor_buff("decoder_memory_bank", step, beam_indices, decoder_hiddens)
            update_tensor_buff("predictions", step, beam_indices,_predictions, False)
            update_tensor_buff("coref_indexes", step, beam_indices, _corefs, False)
            update_tensor_buff("decoder_mask", step, beam_indices, _mask, False)


            variables["input_tokens"] = torch.cat([beam_select_1d(variables["input_tokens"], beam_indices),_input_tokens],1)
            variables["pos_tags"] = torch.cat([beam_select_1d(variables["pos_tags"], beam_indices),_pos_tags],1)
            variables["corefs"] = torch.cat([beam_select_1d(variables["corefs"], beam_indices),_corefs],1)

            # print("_input_tokens.shape:", _input_tokens.shape)
            # # print("_input_tokens", _input_tokens)
            # print("variables[input_tokens].shape:", variables["input_tokens"].shape)
            # print("variables[input_tokens]:", variables["input_tokens"])
            # print("beam_buffer[decoder_mask].shape :", beam_buffer["decoder_mask"].shape)
            # print("beam_buffer[decoder_mask] :", beam_buffer["decoder_mask"])

            pred_sequences = []
            for be in range(beam_size):
                sequence = []
                for t in range(variables["input_tokens"].shape[1]):
                    sequence.append(self.vocab.get_token_from_index(variables["input_tokens"][be,t].item(), "decoder_token_ids")) 
                pred_sequences.append(sequence)
            # print("pred_sequences :", pred_sequences)

            # print("beam_buffer[predictions].shape :", beam_buffer["predictions"].shape)
            # print("beam_buffer[predictions] :", beam_buffer["predictions"])
            # print("beam_buffer[scores].shape :", beam_buffer["scores"].shape)
            # print("beam_buffer[scores] :", beam_buffer["scores"])


        for batch_idx, item in enumerate(bucket):
            if len(item) == 0:
                bucket[batch_idx].append(
                    {
                        key: tensor[batch_idx, 0].unsqueeze(0) for key, tensor in beam_buffer.items()
                    }
                )

        return_dict = {}

        for key in bucket[0][-1].keys():
            return_dict[key] = torch.cat(
                [hypos[-1][key] for hypos in bucket],
                dim=0
            )

       
        return_dict["predictions"] = return_dict["predictions"][:, :-1]

        return_dict["predictions"][return_dict["predictions"] == pad_token] = eos_token
       
        return_dict["decoder_mask"] = 1 - return_dict["decoder_mask"]
        return_dict["decoder_mask"] = return_dict["predictions"] != eos_token #return_dict["decoder_mask"][:, :-1]
        # print(  return_dict["predictions"])
        # print(  return_dict["decoder_mask"])

        return_dict["decoder_memory_bank"] = return_dict["decoder_memory_bank"][:, 1:]
        return_dict["coref_indexes"] = return_dict["coref_indexes"][:, :-1]
        return_dict["scores"] = torch.div(return_dict["scores"], return_dict["decoder_mask"].sum(1, keepdim=True).type_as(return_dict["scores"]))

        # for k,v in return_dict.items():
        #     print("return_dict[k]: ", k, " ", v.shape)

        return return_dict

    def decode_with_pointer_generator(self, encoder_inputs,  mask, copy_attention_maps, copy_vocabs, tag_luts, invalid_indexes):

        # [batch_size, 1]
        batch_size = encoder_inputs.size(0)

        tokens = torch.ones(batch_size, 1) * self.vocab.get_token_index(
            DEFAULT_PADDING_TOKEN, "decoder_token_ids")
        pos_tags = torch.ones(batch_size, 1) * self.vocab.get_token_index(
            DEFAULT_OOV_TOKEN, "pos_tags")
        tokens = tokens.type_as(mask).long()
        pos_tags = pos_tags.type_as(tokens)
        corefs = torch.zeros(batch_size, 1).type_as(mask).long()

        decoder_input_history = []
        decoder_outputs = []
        rnn_outputs = []
        copy_attentions = []
        coref_attentions = []
        predictions = []
        coref_indexes = []
        decoder_mask = []

        coref_inputs = []

        # A sparse indicator matrix mapping each node to its index in the dynamic vocab.
        # Here the maximum size of the dynamic vocab is just max_decode_length.
        coref_attention_maps = torch.zeros(
            batch_size, self.max_decode_length, self.max_decode_length + 1).type_as(mask)
        # A matrix D where the element D_{ij} is for instance i the real vocab index of
        # the generated node at the decoding step `i'.
        coref_vocab_maps = torch.zeros(batch_size, self.max_decode_length + 1).type_as(mask).long()

        # coverage = None
        # if self.use_coverage:
        #     coverage = memory_bank.new_zeros(batch_size, 1, memory_bank.size(1))

        target_copy_attentions = []

        for step_i in range(self.max_decode_length):
<<<<<<< HEAD
            # print("\nstep_i: ", step_i)
=======
            print("\nstep_i: ", step_i)
            # print("tokens: ", tokens)
>>>>>>> tmp

            ## Convert tgttokens to transformer_token ids
            tgt_ids =[] 
            tgt_sequences = []
            for b in range(tokens.size(0)):
                sequence = []
                for t in range(tokens.size(1)):
                    token = self.vocab.get_token_from_index(tokens[b,t].item(), "decoder_token_ids")
<<<<<<< HEAD
=======
                    # print("token: ", token)
>>>>>>> tmp
                    result = re.search("-[0-9]", token)
                    if result is not None:
                        s,e = result.span()
                        token = token[:s]
                    if token in self.amrnodes_tot5_tokens.keys():
                        token = self.amrnodes_tot5_tokens[token].strip()
                    else:
                        token = "<unk>"
                    tokenid = self.transformer_tokenizer.convert_tokens_to_ids(token)
                    if tokenid == 2:
                        tokenid = self.transformer_tokenizer.convert_tokens_to_ids("▁"+token)
<<<<<<< HEAD
=======
                    # print("token -> ", token, " id: ", tokenid)
                    # print("token: ", token, " id: ", tokenid)
>>>>>>> tmp
                    sequence.append(tokenid) 
                tgt_sequences.append(sequence)

            tgt_ids = torch.reshape(torch.tensor(tgt_sequences), (batch_size, step_i+1)).cuda()
<<<<<<< HEAD
            print("tokens: ", tokens)
            print("tgt_ids: ", tgt_ids)
=======
            # print("tgt_ids: ", tgt_ids)
>>>>>>> tmp

            outputs = self.transformers(input_ids=encoder_inputs, attention_mask=mask, decoder_input_ids=tgt_ids, output_attentions=True, output_hidden_states=True, return_dict=True)
            memory_bank = outputs.encoder_last_hidden_state
            decoder_hiddens = outputs.last_hidden_state
            
<<<<<<< HEAD
            
=======
>>>>>>> tmp
            _copy_attentions = outputs.decoder_attentions[11] 
            _copy_attentions = torch.sum(_copy_attentions, dim=1)        # B,Ty,Tx

            _coref_attentions = outputs.decoder_attentions[10]  
            _coref_attentions = torch.sum(_coref_attentions, dim=1)      # B,Ty,Ty


            # print("_copy_attentions_before: ", _copy_attentions.shape)    
            # print("_coref_attentions_before: ", _coref_attentions.shape)

            # 3. Run pointer/generator.
            if step_i == 0:
                _coref_attention_maps = coref_attention_maps[:, :step_i + 1]
            else:
                _coref_attention_maps = coref_attention_maps[:, :step_i + 1] #step_i
                decoder_hiddens = decoder_hiddens[:,-1:,:]
                _copy_attentions = _copy_attentions[:, -1:,:]
                _coref_attentions = _coref_attentions[:, -1:] #:-1

            # print("copy_attention_maps: ", copy_attention_maps.shape)
            # print("coref_attention_maps: ", coref_attention_maps.shape)
            # print("_coref_attention_maps: ", _coref_attention_maps.shape)
            # print("decoder_hiddens: ", decoder_hiddens.shape)
            # print("_copy_attentions: ", _copy_attentions.shape)
            # print("_coref_attentions: ", _coref_attentions.shape)

            generator_output = self.generator(
                decoder_hiddens, _copy_attentions, copy_attention_maps,
                _coref_attentions, _coref_attention_maps, invalid_indexes)
            _predictions = generator_output['predictions']

            # 4. Update maps and get the next token input.
            # _tokens, _predictions, _pos_tags, _corefs, _mask = self._update_maps_and_get_next_input(
            _tokens, _predictions, _pos_tags, _corefs, _mask = self._update_maps_and_get_next_input(
                step_i,
                generator_output['predictions'].squeeze(1),
                generator_output['source_dynamic_vocab_size'],
                coref_attention_maps,
                coref_vocab_maps,
                copy_vocabs,
                decoder_mask,
                tag_luts,
                invalid_indexes
            )

            tokens = torch.cat([tokens,_tokens],1)
            pos_tags = torch.cat([pos_tags,_pos_tags],1)
            corefs = torch.cat([corefs,_corefs],1)
            
            # print("_tokens: ", _tokens)
            # print("_pos_tags: ", pos_tags)
            # print("_corefs: ", _corefs)
            print("_predictions: ", _predictions)
            # print("tokens: ", tokens)

            # 5. Update variables.
            decoder_outputs += [decoder_hiddens]
            copy_attentions += [_copy_attentions]
            coref_attentions += [_coref_attentions]

            predictions += [_predictions]
            # Add the coref info for the next input.
            coref_indexes += [_corefs]
            # Add the mask for the next input.
            decoder_mask += [_mask]


        # 6. Do the following chunking for the graph decoding input.
        # Exclude the hidden state for BOS.
        # decoder_input_history = torch.cat(decoder_input_history[1:], dim=1)
        decoder_outputs = torch.cat(decoder_outputs[1:], dim=1)
        # rnn_outputs = torch.cat(rnn_outputs[1:], dim=1)
        # Exclude coref/mask for EOS.
        # TODO: Answer "What if the last one is not EOS?"
        predictions = torch.cat(predictions[:-1], dim=1)
        coref_indexes = torch.cat(coref_indexes[:-1], dim=1)
        decoder_mask = torch.logical_not(torch.cat(decoder_mask[:-1], dim=1))

        # print("predictions: ", predictions)
        # print("decoder_mask: ", decoder_mask)
        return dict(
            # [batch_size, max_decode_length]
            predictions=predictions,
            coref_indexes=coref_indexes,
            decoder_mask=decoder_mask,
            # [batch_size, max_decode_length, hidden_size]
            # decoder_inputs=decoder_input_history,
            decoder_memory_bank=decoder_outputs,
            # decoder_rnn_memory_bank=decoder_outputs, #rnn_outputs,
            # [batch_size, max_decode_length, encoder_length]
            copy_attentions=copy_attentions,
            coref_attentions=coref_attentions
        )

    def _update_maps_and_get_next_input(self, step, predictions, copy_vocab_size, coref_attention_maps, coref_vocab_maps, copy_vocabs, masks,  tag_luts, invalid_indexes):

        """Dynamically update/build the maps needed for copying.

        :param step: the decoding step, int.
        :param predictions: [batch_size]
        :param copy_vocab_size: int.
        :param coref_attention_maps: [batch_size, max_decode_length, max_decode_length]
        :param coref_vocab_maps:  [batch_size, max_decode_length]
        :param copy_vocabs: a list of dynamic vocabs.
        :param masks: a list of [batch_size] tensors indicating whether EOS has been generated.
            if EOS has has been generated, then the mask is `1`.
        :param tag_luts: a dict mapping key to a list of dicts mapping a source token to a POS tag.
        :param invalid_indexes: a dict storing invalid indexes for copying and generation.
        :return:
        """
        vocab_size = self.generator.vocab_size
        batch_size = predictions.size(0)

        batch_index = torch.arange(0, batch_size).type_as(predictions)
        step_index = torch.full_like(predictions, step)

        gen_mask = predictions.lt(vocab_size)
        copy_mask = predictions.ge(vocab_size).mul(predictions.lt(vocab_size + copy_vocab_size))
        coref_mask = predictions.ge(vocab_size + copy_vocab_size)

        # 1. Update coref_attention_maps
        # Get the coref index.
        coref_index = (predictions - vocab_size - copy_vocab_size)
        # Fill the place where copy didn't happen with the current step,
        # which means that the node doesn't refer to any precedent, it refers to itself.
        coref_index.masked_fill_( torch.logical_not(coref_mask), step + 1)
        
        # coref_index.masked_fill_(1 - coref_mask, step + 1)

        coref_attention_maps[batch_index, step_index, coref_index] = 1

        # 2. Compute the next input.
        # coref_predictions have the dynamic vocabulary index, and OOVs are set to zero.
        coref_predictions = (predictions - vocab_size - copy_vocab_size) * coref_mask.long()
        # Get the actual coreferred token's index in the gen vocab.
        coref_predictions = coref_vocab_maps.gather(1, coref_predictions.unsqueeze(1)).squeeze(1)

        # If a token is copied from the source side, we look up its index in the gen vocab.
        copy_predictions = (predictions - vocab_size) * copy_mask.long()
        # print("bef copy_predictions: ", copy_predictions)

        pos_tags = torch.full_like(predictions, self.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'pos_tags'))
        for i, index in enumerate(copy_predictions.tolist()):
            copied_token = copy_vocabs[i].get_token_from_idx(index)
            # print("index: ", index)
            # print("copy_vocabs: ", copy_vocabs)
            # print("copied_token: ", copied_token)
            # print("tag_luts[i]['pos']: ", tag_luts[i]['pos'])
            if copied_token[:1] == "▁":
                copied_token = copied_token[1:]
            if index != 0:
                if copied_token in tag_luts[i]['pos'].keys():
                    pos_tags[i] = self.vocab.get_token_index(
                        tag_luts[i]['pos'][copied_token], 'pos_tags')
                if False: # is_abstract_token(copied_token):
                    invalid_indexes['source_copy'][i].add(index)
            copy_predictions[i] = self.vocab.get_token_index(copied_token, 'decoder_token_ids')
            # print("copied_token: ", copied_token)
            # print("copy_predictions[i]: ", copy_predictions[i])

        for i, index in enumerate(
                (predictions * gen_mask.long() + coref_predictions * coref_mask.long()).tolist()):
            if index != 0:
                token = self.vocab.get_token_from_index(index, 'decoder_token_ids')
                src_token = find_similar_token(token, list(tag_luts[i]['pos'].keys()))
                if src_token is not None:
                    pos_tags[i] = self.vocab.get_token_index(
                        tag_luts[i]['pos'][src_token], 'pos_tag')
                if False: # is_abstract_token(token):
                    invalid_indexes['vocab'][i].add(index)

        # print("aft copy_predictions: ", copy_predictions)
        # print("coref_mask: ",coref_mask)
        # print("copy_mask: ",copy_mask)
        # print("gen_mask: ",gen_mask)
        next_input = coref_predictions * coref_mask.long() + \
                     copy_predictions * copy_mask.long() + \
                     predictions * gen_mask.long()


        # print("coref_mask: ", coref_mask)
        # print("copy_mask: ", copy_mask)
        # print("next_input: ", next_input)

        # 3. Update dynamic_vocab_maps
        # Here we update D_{step} to the index in the standard vocab.
        coref_vocab_maps[batch_index, step_index + 1] = next_input
        # print("next_input: ",next_input)

        # 4. Get the coref-resolved predictions.
        coref_resolved_preds = coref_predictions * coref_mask.long() + predictions * (torch.logical_not(coref_mask)).long()

        # 5. Get the mask for the current generation.
        has_eos = torch.zeros_like(gen_mask)
        if len(masks) != 0:
            has_eos = torch.cat(masks, 1).long().sum(1).gt(0)

        end_id = self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')
        mask = next_input.eq(end_id) | has_eos
        
        return (next_input.unsqueeze(1),
                coref_resolved_preds.unsqueeze(1),
                pos_tags.unsqueeze(1),
                coref_index.unsqueeze(1),
                mask.unsqueeze(1))

    def decode_with_graph_parser(self,  memory_bank, corefs, mask):
        """Predict edges and edge labels between nodes.
        :param decoder_inputs: [batch_size, node_length, embedding_size]
        :param memory_bank: [batch_size, node_length, hidden_size]
        :param corefs: [batch_size, node_length]
        :param mask:  [batch_size, node_length]
        :return a dict of edge_heads and edge_labels.
            edge_heads: [batch_size, node_length]
            edge_labels: [batch_size, node_length]
        """
        memory_bank, _, _, corefs, mask = self.graph_decoder._add_head_sentinel(
            memory_bank, None, None, corefs, mask)
        (edge_node_h, edge_node_m), (edge_label_h, edge_label_m) = self.graph_decoder.encode(memory_bank)
        edge_node_scores = self.graph_decoder._get_edge_node_scores(edge_node_h, edge_node_m, mask.float())

        edge_heads, edge_labels = self.graph_decoder.mst_decode(
            edge_label_h, edge_label_m, edge_node_scores, corefs, mask)
        return dict(
            edge_heads=edge_heads,
            edge_labels=edge_labels
        )

    @classmethod
    def from_params(cls, vocab, params, transformer_tokenizer, amrnodes_tot5_tokens):

        logger.info('Building the STOG Model...')

<<<<<<< HEAD
        # Source attention
        if params['source_attention']['attention_function'] == 'mlp':
            source_attention = MLPAttention(
                decoder_hidden_size= 512, # params['decoder']['hidden_size'],
                encoder_hidden_size= 512, # params['encoder']['hidden_size'], #* 2,
                attention_hidden_size=params['decoder']['hidden_size'],
                coverage=params['source_attention'].get('coverage', False)
            )
        else:
            source_attention = DotProductAttention(
                decoder_hidden_size=params['decoder']['hidden_size'],
                encoder_hidden_size=params['encoder']['hidden_size'],# * 2,
                share_linear=params['source_attention'].get('share_linear', False)
            )

        source_attention_layer = GlobalAttention(
            decoder_hidden_size= 512, #params['decoder']['hidden_size'],
            encoder_hidden_size= 512, #params['encoder']['hidden_size'], #* 2,
            attention=source_attention
        )

        
=======
>>>>>>> tmp
        switch_input_size = 512 #params['encoder']['hidden_size'] #* 2
        generator = PointerGenerator(
            input_size=512, #params['decoder']['hidden_size'],
            switch_input_size=switch_input_size,
            vocab_size=vocab.get_vocab_size('decoder_token_ids'),
            force_copy=params['generator'].get('force_copy', True),
            vocab_pad_idx=0
        )

        graph_decoder = DeepBiaffineGraphDecoder.from_params(vocab, params['graph_decoder'])

        # Vocab
        punctuation_ids = []
        oov_id = vocab.get_token_index(DEFAULT_OOV_TOKEN, 'decoder_token_ids')
        for c in ',.?!:;"\'-(){}[]':
            c_id = vocab.get_token_index(c, 'decoder_token_ids')
            if c_id != oov_id:
                punctuation_ids.append(c_id)

        logger.info('encoder_token: %d' % vocab.get_vocab_size('encoder_token_ids'))
        logger.info('encoder_chars: %d' % vocab.get_vocab_size('encoder_token_characters'))
        logger.info('decoder_token: %d' % vocab.get_vocab_size('decoder_token_ids'))
        logger.info('decoder_chars: %d' % vocab.get_vocab_size('decoder_token_characters'))

        # v1
        # transformers_config = T5Config(num_layers=3)  
        # transformers = T5Model(config=transformers_config)
      
        # v2
        transformers = T5Model.from_pretrained("t5-small")
        transformers.resize_token_embeddings(len(transformer_tokenizer))
        transformer_tokenizer.save_pretrained("t5-vocab") 
        # print("t5vocab: ",transformer_tokenizer.get_vocab())

        # v3
        # transformers = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')


        return cls(
            vocab=vocab,
            punctuation_ids=punctuation_ids,
            use_char_cnn=params['use_char_cnn'],
            # use_coverage=params['use_coverage'],
            max_decode_length=params.get('max_decode_length', 50),
            transformers=transformers,
            transformer_tokenizer=transformer_tokenizer,
            amrnodes_tot5_tokens = amrnodes_tot5_tokens,
            generator=generator,
            graph_decoder=graph_decoder,
            test_config=params.get('mimick_test', None)
        )
