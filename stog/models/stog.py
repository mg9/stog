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
import math, os

from transformers import T5Tokenizer, T5Model, T5Config
from transformers import EncoderDecoderModel

logger = init_logger()


def character_tensor_from_token_tensor(token_tensor, vocab, character_tokenizer, namespace=dict(tokens="decoder_token_ids", characters="decoder_token_characters")):
    token_str = [vocab.get_token_from_index(i, namespace["tokens"]) for i in token_tensor.view(-1).tolist()]
    max_char_len = max([len(token) for token in token_str])
    indices = []
    for token in token_str:
        token_indices = [vocab.get_token_index(vocab._padding_token) for _ in range(max_char_len)]
        for char_i, character in enumerate(character_tokenizer.tokenize(token)):
            index = vocab.get_token_index(character.text, namespace["characters"])
            token_indices[char_i] = index
        indices.append(token_indices)

    return torch.tensor(indices).view(token_tensor.size(0), token_tensor.size(1), -1).type_as(token_tensor)


class STOG(Model):

    def __init__(self,
                 vocab,
                 punctuation_ids,
                 # use_must_copy_embedding,
                 use_char_cnn,
                 # use_coverage,
                 # use_aux_encoder,
                 # use_bert,
                 max_decode_length,
                 # # Encoder
                 # bert_encoder,
                 # encoder_token_embedding,
                 # encoder_pos_embedding,
                 # encoder_must_copy_embedding,
                 # encoder_char_embedding,
                 # encoder_char_cnn,
                 # encoder_embedding_dropout,
                 # encoder,
                 # encoder_output_dropout,
                 # Decoder
                 decoder_token_embedding,
                 decoder_pos_embedding,
                 decoder_coref_embedding,
                 decoder_char_embedding,
                 decoder_char_cnn,
                 decoder_embedding_dropout,
                 # decoder,
                 # # Aux Encoder
                 # aux_encoder,
                 # aux_encoder_output_dropout,
                 # Transformers
                 transformers,
                 # transformers_attention_layer,
                 transformer_tokenizer,
                 # Generator
                 generator,
                 # Graph decoder
                 graph_decoder,
                 test_config
                 ):
        super(STOG, self).__init__()

        self.vocab = vocab
        self.punctuation_ids = punctuation_ids
        # self.use_must_copy_embedding = use_must_copy_embedding
        self.use_char_cnn = use_char_cnn
        # self.use_coverage = use_coverage
        # self.use_aux_encoder = use_aux_encoder
        # self.use_bert = use_bert
        self.max_decode_length = max_decode_length

        # self.bert_encoder = bert_encoder

        # self.encoder_token_embedding = encoder_token_embedding
        # self.encoder_pos_embedding = encoder_pos_embedding
        # self.encoder_must_copy_embedding = encoder_must_copy_embedding
        # self.encoder_char_embedding = encoder_char_embedding
        # self.encoder_char_cnn = encoder_char_cnn
        # self.encoder_embedding_dropout = encoder_embedding_dropout
        # self.encoder = encoder
        # self.encoder_output_dropout = encoder_output_dropout

        self.decoder_token_embedding = decoder_token_embedding
        self.decoder_pos_embedding = decoder_pos_embedding
        self.decoder_coref_embedding = decoder_coref_embedding
        self.decoder_char_embedding = decoder_char_embedding
        self.decoder_char_cnn = decoder_char_cnn
        self.decoder_embedding_dropout = decoder_embedding_dropout
        # self.decoder = decoder

        # self.aux_encoder = aux_encoder
        # self.aux_encoder_output_dropout = aux_encoder_output_dropout

        self.transformers = transformers
        # self.transformers_attention_layer = transformers_attention_layer
        self.transformer_tokenizer = transformer_tokenizer

        self.generator = generator

        self.graph_decoder = graph_decoder

        self.beam_size = 1

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
        dataset_reader = load_dataset_reader('AMR', word_splitter=word_splitter, transformer_tokenizer= self.transformer_tokenizer)
        dataset_reader.set_evaluation()
        predictor = Predictor.by_name('STOG')(self, dataset_reader)
        manager = _PredictManager(
            predictor,
            self.test_config['data'],
            self.test_config['prediction'],
            self.test_config['batch_size'],
            False,
            True,
            1
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

    def print_batch_details(self, batch, batch_idx):
        print(batch["amr"][batch_idx])
        print()

        print("Source tokens:")
        print([(i, x) for i, x in enumerate(batch["src_tokens_str"][batch_idx])])
        print()

        print('Source copy vocab')
        print(batch["src_copy_vocab"][batch_idx])
        print()

        print('Source map')
        print(batch["src_copy_map"][batch_idx].int())
        print()

        print("Target tokens")
        print([(i, x) for i, x in enumerate(batch["tgt_tokens_str"][batch_idx])])
        print()

        print('Source copy indices')
        print([(i, x) for i, x in enumerate(batch["src_copy_indices"][batch_idx].tolist())])

        print('Target copy indices')
        print([(i, x) for i, x in enumerate(batch["tgt_copy_indices"][batch_idx].tolist())])

    def prepare_batch_input(self, batch):

        # [batch, num_tokens]
        bert_token_inputs = batch.get('src_token_ids', None)
        if bert_token_inputs is not None:
            bert_token_inputs = bert_token_inputs.long()
        encoder_token_subword_index = batch.get('src_token_subword_index', None)
        if encoder_token_subword_index is not None:
            encoder_token_subword_index = encoder_token_subword_index.long()
        encoder_token_inputs =  batch['src_ids'] #batch['src_tokens']['encoder_tokens']
        # encoder_pos_tags = batch['src_pos_tags']
        # encoder_must_copy_tags = batch['src_must_copy_tags']
        # [batch, num_tokens, num_chars]
        # encoder_char_inputs = batch['src_tokens']['encoder_characters']
        # [batch, num_tokens]

        # encoder_mask = get_text_field_mask(batch['src_ids'])
        encoder_mask = (batch['src_ids'] != 0).long()

        encoder_inputs = dict(
            bert_token=bert_token_inputs,
            token_subword_index=encoder_token_subword_index,
            token=encoder_token_inputs,
            # pos_tag=encoder_pos_tags,
            # must_copy_tag=encoder_must_copy_tags,
            # char=encoder_char_inputs,
            mask=encoder_mask
        )

        # [batch, num_tokens]
        decoder_token_inputs = batch['tgt_tokens']['decoder_tokens'][:, :-1].contiguous()
        # decoder_pos_tags = batch['tgt_pos_tags'][:, :-1]
        # [batch, num_tokens, num_chars]
        decoder_char_inputs = batch['tgt_tokens']['decoder_characters'][:, :-1].contiguous()
        # TODO: The following change can be done in amr.py.
        # Initially, raw_coref_inputs has value like [0, 0, 0, 1, 0]
        # where '0' indicates that the input token has no precedent, and
        # '1' indicates that the input token's first precedent is at position '1'.
        # Here, we change it to [0, 1, 2, 1, 4] which means if the input token
        # has no precedent, then it is referred to itself.
        raw_coref_inputs = batch["tgt_copy_indices"][:, :-1].contiguous()
        coref_happen_mask = raw_coref_inputs.ne(0)
        decoder_coref_inputs = torch.ones_like(raw_coref_inputs) * torch.arange(
            0, raw_coref_inputs.size(1)).type_as(raw_coref_inputs).unsqueeze(0)
        decoder_coref_inputs.masked_fill_(coref_happen_mask, 0)
        # [batch, num_tokens]
        decoder_coref_inputs = decoder_coref_inputs + raw_coref_inputs

        decoder_inputs = dict(
            token=decoder_token_inputs,
            # pos_tag=decoder_pos_tags,
            char=decoder_char_inputs,
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

        # print("copy_targets:", copy_targets)
        # print("coref_targets:", coref_targets)
        # print("vocab_targets:", vocab_targets)
        
        # [batch, num_src_tokens + unk, src_dynamic_vocab_size]
        # Exclude the last pad.
        copy_attention_maps = batch['src_copy_map'][:, 1:-1]

        generator_inputs = dict(
            vocab_targets=vocab_targets,
            coref_targets=coref_targets,
            coref_attention_maps=coref_attention_maps,
            copy_targets=copy_targets,
            copy_attention_maps=copy_attention_maps
        )

        # Remove the last two pads so that they have the same size of other inputs?
        edge_heads = batch['head_indices'][:, :-2]
        edge_labels = batch['head_tags'][:, :-2]
        # TODO: The following computation can be done in amr.py.
        # Get the parser mask.
        parser_token_inputs = torch.zeros_like(decoder_token_inputs)
        parser_token_inputs.copy_(decoder_token_inputs)
        parser_token_inputs[
            parser_token_inputs == self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')
        ] = 0
        parser_mask = (parser_token_inputs != 0).float()

        parser_inputs = dict(
            edge_heads=edge_heads,
            edge_labels=edge_labels,
            corefs=decoder_coref_inputs,
            mask=parser_mask
        )
        # import pdb; pdb.set_trace()

        return encoder_inputs, decoder_inputs, generator_inputs, parser_inputs

    def forward(self, batch, for_training=False):
        encoder_inputs, decoder_inputs, generator_inputs, parser_inputs = self.prepare_batch_input(batch)

        if for_training:
            self.transformers.train()

            enc_dec_outputs = self.encode_decode(encoder_inputs['bert_token'],
                                encoder_inputs['token_subword_index'],
                                encoder_inputs['token'],
                                decoder_inputs['token'],
                                # encoder_inputs['pos_tag'],
                                # encoder_inputs['must_copy_tag'],
                                # encoder_inputs['char'],
                                encoder_inputs['mask'],
                                parser_inputs['mask'],
                                # decoder_inputs['pos_tag'],
                                decoder_inputs['char'],
                                decoder_inputs['coref'],
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
                parser_inputs['mask'],
                None
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

            invalid_indexes = dict(
                source_copy=batch.get('source_copy_invalid_ids', None),
                vocab= None #[set(self.punctuation_ids) for _ in range(len(batch['tag_lut']))]
            )

            return dict(
                src_tokens = encoder_inputs['token'],
                encoder_mask=encoder_inputs['mask'],
                copy_attention_maps=generator_inputs['copy_attention_maps'],
                copy_vocabs=batch['src_copy_vocab'],
                # tag_luts=batch['tag_lut'],
                invalid_indexes=invalid_indexes
                # pos_tags=encoder_inputs['pos_tag'],
                # must_copy_tags=encoder_inputs['must_copy_tag'],
                # chars=encoder_inputs['char']
            )

    # def encode_decode(self, bert_tokens, token_subword_index, src_tokens, tgt_tokens, pos_tags, must_copy_tags, chars, mask, tgt_mask, tgt_pos_tags, tgt_chars,  tgt_corefs):
    def encode_decode(self, bert_tokens, token_subword_index, src_tokens, tgt_tokens,  mask, tgt_mask,  tgt_chars,  tgt_corefs):
       
        ### Disable paper raw embeddings for now
        # # [batch, num_tokens, embedding_size]
        # encoder_inputs = []
        # if self.use_bert:
        #     bert_mask = bert_tokens.ne(0)
        #     bert_embeddings, _ = self.bert_encoder(
        #         bert_tokens,
        #         attention_mask=bert_mask,
        #         output_all_encoded_layers=False,
        #         token_subword_index=token_subword_index
        #     )
        #     if token_subword_index is None:
        #         bert_embeddings = bert_embeddings[:, 1:-1]
        #     encoder_inputs += [bert_embeddings]

        # token_embeddings = self.encoder_token_embedding(src_tokens)
        # pos_tag_embeddings = self.encoder_pos_embedding(pos_tags)
        # encoder_inputs += [token_embeddings, pos_tag_embeddings]

        # if self.use_must_copy_embedding:
        #     must_copy_tag_embeddings = self.encoder_must_copy_embedding(must_copy_tags)
        #     encoder_inputs += [must_copy_tag_embeddings]

        # if self.use_char_cnn:
        #     char_cnn_output = self._get_encoder_char_cnn_output(chars)
        #     encoder_inputs += [char_cnn_output]

        # encoder_inputs = torch.cat(encoder_inputs, 2)
        # encoder_inputs = self.encoder_embedding_dropout(encoder_inputs)


        ## Use transformer input (tokenization & embeddings)

        # print("src_tokens: ", src_tokens)
        # print("tgt_tokens: ", tgt_tokens)

        # B,Tx = src_tokens.shape
        # B,Ty = tgt_tokens.shape

        # # Convert srctokens to words
        # src_sequences = []
        # for b in range(B):
        #     sequence = []
        #     for t in range(Tx):
        #         token = self.vocab.get_token_from_index(src_tokens[b,t].item(), "encoder_token_ids")
        #         sequence.append(token) 
        #         if token == END_SYMBOL:
        #             break
        #     src_sequences.append(sequence)
        # print("src_sequences:", src_sequences)

        # #Convert tgttokens to words
        # tgt_sequences = []
        # for b in range(B):
        #     sequence = []
        #     for t in range(Ty):
        #         token = self.vocab.get_token_from_index(tgt_tokens[b,t].item(), "decoder_token_ids")
        #         sequence.append(token) 
        #         if token == END_SYMBOL:
        #             break
        #     if sequence[-1] != END_SYMBOL: # TODO: check this part! Why no </s> at the last sequence
        #         sequence.append(END_SYMBOL) 
        #     tgt_sequences.append(sequence)
        # print("tgt_sequences: ", tgt_sequences)

        # src_train = [" ".join(text).strip() for text in src_sequences]
        # # tgt_train = [" ".join(text).strip() for text in tgt_sequences]
     
        # ## tokenize src 
        # src_train_dict = self.transformer_tokenizer.batch_encode_plus(src_train, pad_to_max_length=True, return_tensors="pt")
       
        # # ## tokenize tgt 
        # # tgt_train_dict = self.transformer_tokenizer.batch_encode_plus(tgt_train, pad_to_max_length=True, return_tensors="pt")
   
        # ## obtain input tensors
        # input_ids = src_train_dict["input_ids"].cuda()
        # # output_ids = tgt_train_dict["input_ids"].cuda()
        
        # ## obtain masks for paddings
        # src_attention_mask = src_train_dict["attention_mask"].cuda()
        # # tgt_attention_mask = tgt_train_dict["attention_mask"].cuda()
        

        # print("input_ids: ", input_ids)
        # # print("output_ids: ", output_ids)
        # print("src_attention_mask: ", src_attention_mask)
        # # print("tgt_attention_mask: ", tgt_attention_mask)

        ## Disable paper raw embeddings for now
        ## Decoder inputs
        # [batch, num_tokens, embedding_size]
        
        token_embeddings = self.decoder_token_embedding(tgt_tokens)
        # pos_tag_embeddings = self.decoder_pos_embedding(tgt_pos_tags)
        coref_embeddings = self.decoder_coref_embedding(tgt_corefs)
        if self.use_char_cnn:
            char_cnn_output = self._get_decoder_char_cnn_output(tgt_chars)
            # decoder_inputs = torch.cat([
            #     token_embeddings, pos_tag_embeddings, coref_embeddings, char_cnn_output], 2)
            decoder_inputs = torch.cat([
                token_embeddings, coref_embeddings, char_cnn_output], 2)
        else:
            # decoder_inputs = torch.cat([
            #     token_embeddings, pos_tag_embeddings, coref_embeddings], 2)

            decoder_inputs = torch.cat([
                 token_embeddings,  coref_embeddings], 2)
        decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

        ### Disable giving our own embeds for now
        # outputs = self.transformers(inputs_embeds=encoder_inputs, decoder_inputs_embeds= decoder_inputs, attention_mask=mask, decoder_attention_mask=tgt_mask, output_attentions=True, output_hidden_states=True, return_dict=True)

        outputs = self.transformers(input_ids=src_tokens, decoder_inputs_embeds=decoder_inputs, attention_mask=mask, decoder_attention_mask=tgt_mask, output_attentions=True, output_hidden_states=True, return_dict=True)
        encoder_hiddens = outputs.encoder_last_hidden_state
        decoder_hiddens = outputs.last_hidden_state

        # attn_h, copy_attentions, coverage = self.transformers_attention_layer(decoder_hiddens, encoder_hiddens, mask)
        coref_attentions = outputs.decoder_attentions[10] 
        coref_attentions = torch.sum(coref_attentions, dim=1)        # B,Ty,Ty
        
        copy_attentions = outputs.decoder_attentions[11] 
        copy_attentions = torch.sum(copy_attentions, dim=1)        # B,Ty,Tx

        # print("encoder_hiddens: ",  encoder_hiddens.shape)
        # print("decoder_hiddens: ",  decoder_hiddens.shape)
        # print("copy_attentions: ",  copy_attentions.shape)
        # print("coref_attentions: ", coref_attentions.shape)

        return dict(
            decoder_hiddens= decoder_hiddens,                              # torch.Size([B, Ty, H])
            coref_attentions=coref_attentions,                             # torch.Size([B, Ty, Ty])
            copy_attentions=copy_attentions                                # torch.Size([B, Ty, Tx])
        )

    def graph_decode(self, memory_bank, edge_heads, edge_labels, corefs, mask, aux_memory_bank):
        # Exclude the BOS symbol.
        memory_bank = memory_bank[:, 1:]
        # if self.use_aux_encoder:
        #     memory_bank = torch.cat([memory_bank, aux_memory_bank], 2)
        corefs = corefs[:, 1:]
        mask = mask[:, 1:]
        return self.graph_decoder(memory_bank, edge_heads, edge_labels, corefs, mask)

    def _get_encoder_char_cnn_output(self, chars):
        # [batch, num_tokens, num_chars, embedding_size]
        char_embeddings = self.encoder_char_embedding(chars)
        batch_size, num_tokens, num_chars, _ = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
        char_cnn_output = self.encoder_char_cnn(char_embeddings, None)
        char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)
        return char_cnn_output

    def _get_decoder_char_cnn_output(self, chars):
        # [batch, num_tokens, num_chars, embedding_size]
        char_embeddings = self.decoder_char_embedding(chars)
        batch_size, num_tokens, num_chars, _ = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
        char_cnn_output = self.decoder_char_cnn(char_embeddings, None)
        char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)
        return char_cnn_output

    def decode(self, input_dict):

        src_tokens = input_dict['src_tokens']
        mask = input_dict['encoder_mask']
        copy_attention_maps = input_dict['copy_attention_maps']
        copy_vocabs = input_dict['copy_vocabs']
        # tag_luts = input_dict['tag_luts']
        invalid_indexes = input_dict['invalid_indexes']
       
        # pos_tags=input_dict['pos_tags']
        # must_copy_tags=input_dict['must_copy_tags']
        # chars=input_dict['chars']
        
        ### Disable giving our own embeds for now
        # encoder_inputs = []
        # token_embeddings = self.encoder_token_embedding(src_tokens)
        # pos_tag_embeddings = self.encoder_pos_embedding(pos_tags)
        # encoder_inputs += [token_embeddings, pos_tag_embeddings]

        # if self.use_must_copy_embedding:
        #     must_copy_tag_embeddings = self.encoder_must_copy_embedding(must_copy_tags)
        #     encoder_inputs += [must_copy_tag_embeddings]

        # if self.use_char_cnn:
        #     char_cnn_output = self._get_encoder_char_cnn_output(chars)
        #     encoder_inputs += [char_cnn_output]

        # encoder_inputs = torch.cat(encoder_inputs, 2)
        # encoder_inputs = self.encoder_embedding_dropout(encoder_inputs) # dropout at predict ?
        # generator_outputs = self.decode_with_pointer_generator(encoder_inputs, mask, copy_attention_maps, copy_vocabs, tag_luts, invalid_indexes)
        
        generator_outputs = self.decode_with_pointer_generator(src_tokens, mask, copy_attention_maps, copy_vocabs, invalid_indexes)

        parser_outputs = self.decode_with_graph_parser(
            generator_outputs['decoder_rnn_memory_bank'],
            generator_outputs['coref_indexes'],
            generator_outputs['decoder_mask']
        )

        return dict(
            nodes= generator_outputs['predictions'],
            heads=parser_outputs['edge_heads'],
            head_labels=parser_outputs['edge_labels'],
            corefs=generator_outputs['coref_indexes']
        )

    # def decode_with_pointer_generator(self, encoder_inputs, mask, copy_attention_maps, copy_vocabs, tag_luts, invalid_indexes):
    def decode_with_pointer_generator(self, encoder_inputs, mask, copy_attention_maps, copy_vocabs, invalid_indexes):

        # [batch_size, 1]
        batch_size = encoder_inputs.size(0)

        tokens = torch.ones(batch_size, 1) * self.vocab.get_token_index(
            START_SYMBOL, "decoder_token_ids")
        # pos_tags = torch.ones(batch_size, 1) * self.vocab.get_token_index(
        #     DEFAULT_OOV_TOKEN, "pos_tags")
        tokens = tokens.type_as(mask).long()
        # pos_tags = pos_tags.type_as(tokens)
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
            # print("\nstep_i: ", step_i)
            # 1. Get the decoder inputs.
            token_embeddings = self.decoder_token_embedding(tokens)
            # pos_tag_embeddings = self.decoder_pos_embedding(pos_tags)
            coref_embeddings = self.decoder_coref_embedding(corefs)

            if self.use_char_cnn:
                # TODO: get chars from tokens.
                # [batch_size, 1, num_chars]
                chars = character_tensor_from_token_tensor(
                    tokens,
                    self.vocab,
                    self.character_tokenizer
                )

                char_cnn_output = self._get_decoder_char_cnn_output(chars)
                # decoder_inputs = torch.cat(
                #     [token_embeddings, pos_tag_embeddings,
                #      coref_embeddings, char_cnn_output], 2)

                decoder_inputs = torch.cat(
                    [token_embeddings, coref_embeddings, char_cnn_output], 2)

            else:
                # decoder_inputs = torch.cat(
                #     [token_embeddings, pos_tag_embeddings, coref_embeddings], 2)
                decoder_inputs = torch.cat(
                     [token_embeddings,  coref_embeddings], 2)

            outputs = self.transformers(input_ids=encoder_inputs, attention_mask=mask, decoder_inputs_embeds=decoder_inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
            memory_bank = outputs.encoder_last_hidden_state
            decoder_hiddens = outputs.last_hidden_state
            
            # attn_h, _copy_attentions, coverage = self.transformers_attention_layer(decoder_hiddens, memory_bank, mask)
            
            _copy_attentions = outputs.decoder_attentions[11] 
            _copy_attentions = torch.sum(_copy_attentions, dim=1)        # B,Ty,Tx

            _coref_attentions = outputs.decoder_attentions[10]  
            _coref_attentions = torch.sum(_coref_attentions, dim=1)      # B,Ty,Ty


            # 3. Run pointer/generator.
            if step_i == 0:
                _coref_attention_maps = coref_attention_maps[:, :step_i + 1]
            else:
                _coref_attention_maps = coref_attention_maps[:, :step_i]
                decoder_hiddens = decoder_hiddens[:,-1:,:]
                _copy_attentions = _copy_attentions[:, -1:,:]
                _coref_attentions = _coref_attentions[:, -1:, :-1]

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
            _tokens, _predictions,  _corefs, _mask = self._update_maps_and_get_next_input(
                step_i,
                generator_output['predictions'].squeeze(1),
                generator_output['source_dynamic_vocab_size'],
                coref_attention_maps,
                coref_vocab_maps,
                copy_vocabs,
                decoder_mask,
                # tag_luts,
                invalid_indexes
            )

            tokens = torch.cat([tokens,_tokens],1)
            # pos_tags = torch.cat([pos_tags,_pos_tags],1)
            corefs = torch.cat([corefs,_corefs],1)
            
            # print("_tokens: ", _tokens)
            # print("_pos_tags: ", pos_tags)
            # print("_corefs: ", _corefs)
            # print("_predictions: ", _predictions)
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

        return dict(
            # [batch_size, max_decode_length]
            predictions=predictions,
            coref_indexes=coref_indexes,
            decoder_mask=decoder_mask,
            # [batch_size, max_decode_length, hidden_size]
            # decoder_inputs=decoder_input_history,
            decoder_memory_bank=decoder_outputs,
            decoder_rnn_memory_bank=decoder_outputs, #rnn_outputs,
            # [batch_size, max_decode_length, encoder_length]
            copy_attentions=copy_attentions,
            coref_attentions=coref_attentions
        )

    # def _update_maps_and_get_next_input(self, step, predictions, copy_vocab_size, coref_attention_maps, coref_vocab_maps, copy_vocabs, masks, tag_luts, invalid_indexes):
    def _update_maps_and_get_next_input(self, step, predictions, copy_vocab_size, coref_attention_maps, coref_vocab_maps, copy_vocabs, masks,  invalid_indexes):

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

        coref_attention_maps[batch_index, step_index, coref_index] = 1

        # 2. Compute the next input.
        # coref_predictions have the dynamic vocabulary index, and OOVs are set to zero.
        coref_predictions = (predictions - vocab_size - copy_vocab_size) * coref_mask.long()
        # Get the actual coreferred token's index in the gen vocab.
        coref_predictions = coref_vocab_maps.gather(1, coref_predictions.unsqueeze(1)).squeeze(1)

        # If a token is copied from the source side, we look up its index in the gen vocab.
        copy_predictions = (predictions - vocab_size) * copy_mask.long()
        # pos_tags = torch.full_like(predictions, self.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'pos_tags'))
        for i, index in enumerate(copy_predictions.tolist()):
            copied_token = copy_vocabs[i].get_token_from_idx(index)
            # if index != 0:
            #     pos_tags[i] = self.vocab.get_token_index(
            #         tag_luts[i]['pos'][copied_token], 'pos_tags')
            #     if False: # is_abstract_token(copied_token):
            #         invalid_indexes['source_copy'][i].add(index)
            copy_predictions[i] = self.vocab.get_token_index(copied_token, 'decoder_token_ids')

        for i, index in enumerate(
                (predictions * gen_mask.long() + coref_predictions * coref_mask.long()).tolist()):
            if index != 0:
                token = self.vocab.get_token_from_index(index, 'decoder_token_ids')
                # src_token = find_similar_token(token, list(tag_luts[i]['pos'].keys()))
                # if src_token is not None:
                #     pos_tags[i] = self.vocab.get_token_index(
                #         tag_luts[i]['pos'][src_token], 'pos_tag')
                # if False: # is_abstract_token(token):
                #     invalid_indexes['vocab'][i].add(index)

        next_input = coref_predictions * coref_mask.long() + \
                     copy_predictions * copy_mask.long() + \
                     predictions * gen_mask.long()

        # 3. Update dynamic_vocab_maps
        # Here we update D_{step} to the index in the standard vocab.
        coref_vocab_maps[batch_index, step_index + 1] = next_input

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
                # pos_tags.unsqueeze(1),
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
    def from_params(cls, vocab, params, transformer_tokenizer):

        logger.info('Building the STOG Model...')

        ### Disable paper encoder for now
        # # Encoder
        # encoder_input_size = 0
        # bert_encoder = None
        # if params.get('use_bert', False):
        #     bert_encoder = Seq2SeqBertEncoder.from_pretrained(params['bert']['pretrained_model_dir'])
        #     encoder_input_size += params['bert']['hidden_size']
        #     for p in bert_encoder.parameters():
        #         p.requires_grad = False

        # encoder_token_embedding = Embedding.from_params(vocab, params['encoder_token_embedding'])
        # encoder_input_size += params['encoder_token_embedding']['embedding_dim']
        # encoder_pos_embedding = Embedding.from_params(vocab, params['encoder_pos_embedding'])
        # encoder_input_size += params['encoder_pos_embedding']['embedding_dim']

        # encoder_must_copy_embedding = None
        # if params.get('use_must_copy_embedding', False):
        #     encoder_must_copy_embedding = Embedding.from_params(
        #     vocab, params['encoder_must_copy_embedding'])
        #     encoder_input_size += params['encoder_must_copy_embedding']['embedding_dim']

        # if params['use_char_cnn']:
        #     encoder_char_embedding = Embedding.from_params(vocab, params['encoder_char_embedding'])
        #     encoder_char_cnn = CnnEncoder(
        #         embedding_dim=params['encoder_char_cnn']['embedding_dim'],
        #         num_filters=params['encoder_char_cnn']['num_filters'],
        #         ngram_filter_sizes=params['encoder_char_cnn']['ngram_filter_sizes'],
        #         conv_layer_activation=torch.tanh
        #     )
        #     encoder_input_size += params['encoder_char_cnn']['num_filters']
        # else:
        #     encoder_char_embedding = None
        #     encoder_char_cnn = None

        # encoder_embedding_dropout = InputVariationalDropout(p=params['encoder_token_embedding']['dropout'])

        # params['encoder']['input_size'] = encoder_input_size
        # encoder = PytorchSeq2SeqWrapper(
        #     module=StackedBidirectionalLstm.from_params(params['encoder']),
        #     stateful=True
        # )
        # encoder_output_dropout = InputVariationalDropout(p=params['encoder']['dropout'])


        ### Disable paper decoder for now
        # Decoder
        decoder_input_size = params['decoder']['hidden_size']
        decoder_input_size += params['decoder_token_embedding']['embedding_dim']
        decoder_input_size += params['decoder_coref_embedding']['embedding_dim']
        decoder_input_size += params['decoder_pos_embedding']['embedding_dim']
        decoder_token_embedding = Embedding.from_params(vocab, params['decoder_token_embedding'])
        decoder_coref_embedding = Embedding.from_params(vocab, params['decoder_coref_embedding'])
        decoder_pos_embedding = Embedding.from_params(vocab, params['decoder_pos_embedding'])
        if params['use_char_cnn']:
            decoder_char_embedding = Embedding.from_params(vocab, params['decoder_char_embedding'])
            decoder_char_cnn = CnnEncoder(
                embedding_dim=params['decoder_char_cnn']['embedding_dim'],
                num_filters=params['decoder_char_cnn']['num_filters'],
                ngram_filter_sizes=params['decoder_char_cnn']['ngram_filter_sizes'],
                conv_layer_activation=torch.tanh
            )
            decoder_input_size += params['decoder_char_cnn']['num_filters']
        else:
            decoder_char_embedding = None
            decoder_char_cnn = None

        decoder_embedding_dropout = InputVariationalDropout(p=params['decoder_token_embedding']['dropout'])

        # # Coref attention
        # if params['coref_attention']['attention_function'] == 'mlp':
        #     coref_attention = MLPAttention(
        #         decoder_hidden_size=params['decoder']['hidden_size'],
        #         encoder_hidden_size=params['decoder']['hidden_size'],
        #         attention_hidden_size=params['decoder']['hidden_size'],
        #         coverage=params['coref_attention'].get('coverage', False),
        #         use_concat=params['coref_attention'].get('use_concat', False)
        #     )
        # elif params['coref_attention']['attention_function'] == 'biaffine':
        #     coref_attention = BiaffineAttention(
        #         input_size_decoder=params['decoder']['hidden_size'],
        #         input_size_encoder=params['encoder']['hidden_size'] * 2,
        #         hidden_size=params['coref_attention']['hidden_size']
        #     )
        # else:
        #     coref_attention = DotProductAttention(
        #         decoder_hidden_size=params['decoder']['hidden_size'],
        #         encoder_hidden_size=params['decoder']['hidden_size'],
        #         share_linear=params['coref_attention'].get('share_linear', True)
        #     )

        # coref_attention_layer = GlobalAttention(
        #     decoder_hidden_size=params['decoder']['hidden_size'],
        #     encoder_hidden_size=params['decoder']['hidden_size'],
        #     attention=coref_attention
        # )

        # params['decoder']['input_size'] = decoder_input_size
        # decoder = InputFeedRNNDecoder(
        #     rnn_cell=StackedLstm.from_params(params['decoder']),
        #     attention_layer=source_attention_layer,
        #     coref_attention_layer=coref_attention_layer,
        #     # TODO: modify the dropout so that the dropout mask is unchanged across the steps.
        #     dropout=InputVariationalDropout(p=params['decoder']['dropout']),
        #     use_coverage=params['use_coverage']
        # )

        # if params.get('use_aux_encoder', False):
        #     aux_encoder = PytorchSeq2SeqWrapper(
        #         module=StackedBidirectionalLstm.from_params(params['aux_encoder']),
        #         stateful=True
        #     )
        #     aux_encoder_output_dropout = InputVariationalDropout(
        #         p=params['aux_encoder']['dropout'])
        # else:
        #     aux_encoder = None
        #     aux_encoder_output_dropout = None


        # Source attention
        if params['source_attention']['attention_function'] == 'mlp':
            source_attention = MLPAttention(
                decoder_hidden_size=512, # params['decoder']['hidden_size'],
                encoder_hidden_size=512, # params['encoder']['hidden_size'], #* 2,
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
            decoder_hidden_size=512, #params['decoder']['hidden_size'],
            encoder_hidden_size=512, #params['encoder']['hidden_size'], #* 2,
            attention=source_attention
        )

        
        switch_input_size = 512 #params['encoder']['hidden_size'] #* 2
        generator = PointerGenerator(
            input_size=512,#params['decoder']['hidden_size'],
            switch_input_size=switch_input_size,
            vocab_size=vocab.get_vocab_size('decoder_token_ids'),
            force_copy=params['generator'].get('force_copy', True),
            # TODO: Set the following indices.
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
        # v3
        # transformers = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')


        return cls(
            vocab=vocab,
            punctuation_ids=punctuation_ids,
            # use_must_copy_embedding=params.get('use_must_copy_embedding', False),
            use_char_cnn=params['use_char_cnn'],
            # use_coverage=params['use_coverage'],
            # use_aux_encoder=params.get('use_aux_encoder', False),
            # use_bert=params.get('use_bert', False),
            max_decode_length=params.get('max_decode_length', 50),
            # bert_encoder=bert_encoder,
            # encoder_token_embedding=encoder_token_embedding,
            # encoder_pos_embedding=encoder_pos_embedding,
            # encoder_must_copy_embedding=encoder_must_copy_embedding,
            # encoder_char_embedding=encoder_char_embedding,
            # encoder_char_cnn=encoder_char_cnn,
            # encoder_embedding_dropout=encoder_embedding_dropout,
            # encoder=encoder,
            # encoder_output_dropout=encoder_output_dropout,
            decoder_token_embedding=decoder_token_embedding,
            decoder_coref_embedding=decoder_coref_embedding,
            decoder_pos_embedding=decoder_pos_embedding,
            decoder_char_cnn=decoder_char_cnn,
            decoder_char_embedding=decoder_char_embedding,
            decoder_embedding_dropout=decoder_embedding_dropout,
            # decoder=decoder,
            # aux_encoder=aux_encoder,
            # aux_encoder_output_dropout=aux_encoder_output_dropout,
            transformers=transformers,
            # transformers_attention_layer=source_attention_layer,
            transformer_tokenizer=transformer_tokenizer,
            generator=generator,
            graph_decoder=graph_decoder,
            test_config=params.get('mimick_test', None)
        )
