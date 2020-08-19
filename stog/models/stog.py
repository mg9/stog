import torch
from stog.models.model import Model
from stog.utils.logging import init_logger
from stog.utils.nn import get_text_field_mask
from stog.utils.string import  END_SYMBOL, find_similar_token, is_abstract_token
import numpy as np
import h5py,os
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
logger = init_logger()


class STOG(Model):

    def __init__(self, vocab,  t5, t5_tokenizer):
        super(STOG, self).__init__()

        self.vocab = vocab
        self.t5= t5
        self.t5_tokenizer = t5_tokenizer  


    def prepare_batch_input(self, batch, save=False):
        # [batch, num_tokens]
        bert_token_inputs = batch.get('src_token_ids', None)
        if bert_token_inputs is not None:
            bert_token_inputs = bert_token_inputs.long()
        encoder_token_subword_index = batch.get('src_token_subword_index', None)
        if encoder_token_subword_index is not None:
            encoder_token_subword_index = encoder_token_subword_index.long()
        encoder_token_inputs = batch['src_tokens']['encoder_tokens']
        encoder_pos_tags = batch['src_pos_tags']
        encoder_must_copy_tags = batch['src_must_copy_tags']
        # [batch, num_tokens, num_chars]
        encoder_char_inputs = batch['src_tokens']['encoder_characters']
        # [batch, num_tokens]
        encoder_mask = get_text_field_mask(batch['src_tokens'])

        encoder_inputs = dict(
            bert_token=bert_token_inputs,
            token_subword_index=encoder_token_subword_index,
            token=encoder_token_inputs,
            pos_tag=encoder_pos_tags,
            must_copy_tag=encoder_must_copy_tags,
            char=encoder_char_inputs,
            mask=encoder_mask
        )

        # [batch, num_tokens]
        decoder_token_inputs = batch['tgt_tokens']['decoder_tokens'].contiguous() #[:, :-1].contiguous()
        decoder_pos_tags = batch['tgt_pos_tags'][:, :-1]
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
        

        decoder_coref_inputs.masked_fill_(coref_happen_mask > 0, 0)
        # [batch, num_tokens]
        decoder_coref_inputs = decoder_coref_inputs + raw_coref_inputs

        decoder_inputs = dict(
            token=decoder_token_inputs,
            pos_tag=decoder_pos_tags,
            char=decoder_char_inputs,
            coref=decoder_coref_inputs
        )

        # [batch, num_tokens]
        vocab_targets = batch['tgt_tokens']['decoder_tokens'][:, 1:].contiguous()
        # [batch, num_tokens]
        coref_targets = batch["tgt_copy_indices"][:, 1:]
        # [batch, num_tokens, num_tokens + coref_na]
      

        #print("batch copy map:", batch['tgt_copy_map']) # exclude BOS

        coref_attention_maps = batch['tgt_copy_map'][:, 1:]  # exclude BOS
        # [batch, num_tgt_tokens, num_src_tokens + unk]
        copy_targets = batch["src_copy_indices"][:, 1:]
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
        #print("--")
        #print("edge_heads.shape: ", edge_heads.shape)
        edge_labels = batch['head_tags'][:, :-2]
        # TODO: The following computation can be done in amr.py.
        # Get the parser mask.
        parser_token_inputs = torch.zeros_like(decoder_token_inputs)
        parser_token_inputs.copy_(decoder_token_inputs)
        parser_token_inputs[
            parser_token_inputs == self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')
        ] = 0
        parser_mask = (parser_token_inputs != 0).float()
        #print("parser_mask.shape: ", parser_mask.shape)

        parser_inputs = dict(
            edge_heads=edge_heads,
            edge_labels=edge_labels,
            corefs=decoder_coref_inputs,
            mask=parser_mask
        )

        if save:
            hf.create_dataset('encoder_token_inputs', data=encoder_token_inputs)
            hf.create_dataset('encoder_pos_tags', data=encoder_pos_tags)
            hf.create_dataset('encoder_must_copy_tags', data=encoder_must_copy_tags)
            hf.create_dataset('encoder_char_inputs', data=encoder_char_inputs)
            hf.create_dataset('encoder_mask', data=encoder_mask)
            hf.create_dataset('edge_heads', data=edge_heads)
            hf.create_dataset('edge_labels', data=edge_labels)
            hf.create_dataset('decoder_coref_inputs', data=decoder_coref_inputs)
            hf.create_dataset('parser_mask', data=parser_mask)



        return encoder_inputs, decoder_inputs, generator_inputs, parser_inputs

    def forward(self, batch, for_training=False, save=False):
        encoder_inputs, decoder_inputs, generator_inputs, parser_inputs = self.prepare_batch_input(batch, save=save)
        if for_training:
            self.t5.train()
        else:
            self.t5.eval()
        t5_outputs = self.t5_fw(encoder_inputs['token'], decoder_inputs['token'], for_training=for_training)
        return dict(loss=t5_outputs['loss'])

    def t5_fw(self, src_tokens, tgt_tokens, for_training=False):

        B,Tx = src_tokens.shape
        B,Ty = tgt_tokens.shape

        # Convert srctokens to words
        src_sequences = []
        for b in range(B):
            sequence = []
            for t in range(Tx):
                sequence.append(self.vocab.get_token_from_index(src_tokens[b,t].item(), "encoder_token_ids")) 
            src_sequences.append(sequence)

        # Convert tgttokens to words
        tgt_sequences = []
        for b in range(B):
            sequence = []
            for t in range(Ty):
                sequence.append(self.vocab.get_token_from_index(tgt_tokens[b,t].item(), "decoder_token_ids")) 
            tgt_sequences.append(sequence)
        
   
        src_train = [" ".join(text).strip() for text in src_sequences]
        tgt_train = [" ".join(text).strip() for text in tgt_sequences]
     
        # tokenize src 
        src_train_dict = self.t5_tokenizer.batch_encode_plus(src_train, pad_to_max_length=True, return_tensors="pt")
       
        # tokenize tgt 
        tgt_train_dict = self.t5_tokenizer.batch_encode_plus(tgt_train, pad_to_max_length=True, return_tensors="pt") #max_length=Ty, truncation=True,
   

        # # obtain input tensors
        input_ids = src_train_dict["input_ids"].cuda()
        output_ids = tgt_train_dict["input_ids"].cuda()
        outputs = self.t5(input_ids=input_ids, labels=output_ids)#, use_cache=False, output_attentions=True, output_hidden_states=True)
        loss = outputs[0]
        return dict(loss=loss)

        # if for_training:
        #     #print("for trn...")
        #     pass
        # else:
        #     print("for dev...")
        #     print("src_train: ", src_train)
        #     print("tgt_train: ", tgt_train)
        # print("src_tokens: ", src_tokens)
        # print("tgt_tokens: ", tgt_tokens)
        # print("src_sequences: ", src_sequences)
        # print("tgt_sequences: ", tgt_sequences)
        # print("input_ids: ", input_ids)
        # print("output_ids: ", output_ids)


    @classmethod
    def from_params(cls, vocab, t5_tokenizer, params):
        logger.info('Building the STOG Model...')

        ## Load t5 model
        if os.path.isdir("t5-small-amrtrained"):
            t5 = T5ForConditionalGeneration.from_pretrained("t5-small-amrtrained")
            print("Mylog: Loading the model from t5-small-amrtrained directory...")
        else:
            t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
            print("Mylog: Loading the model from t5-small...")

        t5.resize_token_embeddings(len(t5_tokenizer))
        
        ## Save t5 tokenizer for the first time
        if not os.path.isdir("t5-vocab"):
            t5_tokenizer.save_pretrained("t5-vocab")

        return cls(vocab=vocab, t5= t5, t5_tokenizer = t5_tokenizer)
