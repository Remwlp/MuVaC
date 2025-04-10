import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from transformers import BartTokenizer, BartForConditionalGeneration,BartModel
from torch.nn.utils import weight_norm
import math
import copy
from typing import Optional, List

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers.modeling_utils import PreTrainedModel, unwrap_model

from transformers import (
    BartTokenizerFast,
    AdamW
)
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.models.bart.configuration_bart import BartConfig

from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartDecoder,
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
)
from transformers.models.bart.modeling_bart import shift_tokens_right as my_shift_tokens_right

import random

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput, 
    Seq2SeqSequenceClassifierOutput
)

class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GaussianEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc21 = nn.Linear(output_dim, output_dim)  # mu values
        self.fc22 = nn.Linear(output_dim, output_dim)  # var values
        # setup the non-linearities
        self.act = nn.GELU()



    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self,x):
        hidden = self.act(self.fc1(x))
        z_mean = self.fc21(hidden)
        log_var = self.fc22(hidden)
        z_var = torch.exp(self.fc22(hidden))
        m = self.reparameterize(z_mean, log_var)
        return z_mean, z_var, m
    

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(CrossAttention, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value):
        if query.shape[-1] != 768:
            query = self.text_linear(query)
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(key.size(-1), dtype=torch.float32)
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        attended_values = self.dropout(attended_values)
        return attended_values


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, dim_feedforward=2048):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.encoder = _TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None):
        if lens is not None:
            max_len = max(lens)

            mask = [([False] * l + [True] * (max_len - l)) for l in lens]
            mask = torch.tensor(mask).to(device=inputs.device)
        else:
            mask = None

        inputs = inputs.permute(1, 0, 2)

        inputs = inputs * math.sqrt(self.d_model)
        inputs = self.pos_encoder(inputs)

        outputs = self.encoder(src=inputs, src_key_padding_mask=mask) # (seq_len, bs, dim)

        return [o.permute(1, 0, 2) for o in outputs]

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    
def padTensor(t: torch.Tensor, targetLen: int) -> torch.Tensor:
    oriLen, dim = t.size()
    return torch.cat((t, torch.zeros(targetLen - oriLen, dim).to(t.device)), dim=0)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class _TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = [src]

        for mod in self.layers:
            output = mod(outputs[-1], src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)

        if self.norm is not None:
            outputs[-1] = self.norm(outputs[-1])

        return outputs[1:]

    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




class ContextAwareAttention(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dim_context: int,
                 dropout_rate: Optional[float]=0.0):
        super(ContextAwareAttention, self).__init__()
        
        self.dim_model = dim_model
        self.dim_context = dim_context
        self.dropout_rate = dropout_rate
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model, 
                                                     num_heads=1, 
                                                     dropout=self.dropout_rate, 
                                                     bias=True,
                                                     add_zero_attn=False,
                                                     batch_first=True,
                                                     device=torch.device("cuda")
                                                    )


        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)
        
        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias=False)
        




    def forward(self,
                q: torch.Tensor, 
                k: torch.Tensor,
                v: torch.Tensor,
                context: Optional[torch.Tensor]=None):
        
        key_context = self.u_k(context)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1 - lambda_k) * k + lambda_k * key_context
        v_cap = (1 - lambda_v) * v + lambda_v * value_context

        attention_output, _ = self.attention_layer(query=q,
                                                   key=k_cap,
                                                   value=v_cap)
        return attention_output
         



class BiTransformer(nn.Module):
    def __init__(self, nb_vocab, num_head, d_model, dropout=.1, use_position_emb=False, max_seq_len=18):
        super(BiTransformer, self).__init__()
        self.exp_embed = nn.Embedding(num_embeddings=nb_vocab, embedding_dim=d_model, padding_idx=0)
        self.transformer_layer1 = TransformerEncoder(d_model, 1, num_head,d_model)
        self.transformer_layer2 = TransformerEncoder(d_model, 1, num_head,d_model)
        self.use_position_emb = use_position_emb
        

    def token_emb(self, ids):
        if ids.ndim == 1:
            ids = ids.unsqueeze(1)

        textual_emb = self.exp_embed(ids)
        
        if self.use_position_emb:
            return textual_emb+ self.position_emb[:, :ids.shape[1]]
        else:
            return textual_emb
    

    def forward(self, pre_words):
        word_features = self.token_emb(pre_words)
        word_features = self.transformer_layer1(word_features)[-1]
        word_features = self.transformer_layer2(word_features)[-1]

        return word_features


class AlignAndFusion(nn.Module):
    
    def __init__(self,
                 dim_model: int,
                 dropout_rate=0.1):
        super(AlignAndFusion, self).__init__()
        self.dropout_rate = dropout_rate



        self.align = CrossAttention(dim_model)
        self.acoustic_context_transform = nn.Linear(1, 150, bias=False)     
        self.visual_context_transform = nn.Linear(106, 150, bias=False)
        
        self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                                dim_context=768,
                                                                dropout_rate=dropout_rate)
        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=768,
                                                              dropout_rate=dropout_rate)        
        self.acoustic_gate = nn.Linear(2*dim_model, dim_model)
        self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

    def forward(self,
                text_input: torch.Tensor,
                acoustic_context: Optional[torch.Tensor]=None,
                visual_context: Optional[torch.Tensor]=None):
        aligned_v = self.align(visual_context, text_input, text_input)    # [B, N_t, D_t]
        aligned_a = self.align(acoustic_context, text_input, text_input)    # [B, N_t, D_t]



        visual_context = aligned_v.permute(0,2,1)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0,2,1)

        video_out = self.visual_context_attention(q=text_input,
                                                  k=text_input,
                                                  v=text_input,
                                                  context=visual_context)         
        acoustic_context = aligned_a.permute(0,2,1)
        acoustic_context = self.acoustic_context_transform(acoustic_context)
        acoustic_context = acoustic_context.permute(0,2,1)
        audio_out = self.acoustic_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=acoustic_context)
        
        a_out = self.visual_context_attention(q=text_input,
                                            k=audio_out,
                                            v=audio_out,
                                            context=visual_context)  
        v_out = self.acoustic_context_attention(q=text_input,
                                            k=video_out,
                                            v=video_out,
                                            context=acoustic_context)

        weight_a = F.sigmoid(self.acoustic_gate(torch.cat((a_out, text_input), dim=-1)))
        weight_v = F.sigmoid(self.visual_gate(torch.cat((v_out, text_input), dim=-1)))



        output = self.final_layer_norm(text_input + weight_a * a_out + weight_v * v_out)

        return output
    

class MultimodalBartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False
        

        self.fusion_at_layer5 = [5]
        

        self.MyAAF = AlignAndFusion(768)
        self.Mylinear1 = nn.Linear(768,768,bias=False)
        self.Mylinear2 = nn.Linear(768,768,bias=False)
        self.Mylinear3 = nn.Linear(512,768,bias=False)



        # =============================================================================== #

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of visual_input
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_ids)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            
            if idx in self.fusion_at_layer5:
                    hidden_states = self.Mylinear1(hidden_states)
                    visual_input = self.Mylinear2(visual_input)
                    acoustic_input = self.Mylinear3(acoustic_input)
                    hidden_states = self.MyAAF(hidden_states,acoustic_input,visual_input)
            
            # print(hidden_states.size())
            # =============================================================================== #
            
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                                 

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # essemble_feature = hidden_states

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
# -------------------------------------------------- Multimodal MultimodalBartModel --------------------------------------------------

class MultimodalBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MultimodalBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of visual_input
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = my_shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs= self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                acoustic_input=acoustic_input,      # New addition of acoustic_input
                visual_input=visual_input,      # New addition of visual_input
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# ---------------------------------------------- MultiModalBartForConditionalGeneration ----------------------------------------------
class MultimodalBartForCausalAnswer(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = MultimodalBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of visual_input
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        exp=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if exp is not None:
            if decoder_input_ids is None:
                decoder_input_ids = my_shift_tokens_right(
                    exp, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs_exp = self.model(
            input_ids,
            attention_mask=attention_mask,
            acoustic_input=acoustic_input,      # New addition of acoustic_input
            visual_input=visual_input,      # New addition of visual_input
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs_exp[0]) + self.final_logits_bias

        masked_lm_loss = None
        if exp is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), exp.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs_exp[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs_exp.past_key_values,
                decoder_hidden_states=outputs_exp.decoder_hidden_states,
                decoder_attentions=outputs_exp.decoder_attentions,
                cross_attentions=outputs_exp.cross_attentions,
                encoder_last_hidden_state=outputs_exp.encoder_last_hidden_state,
                encoder_hidden_states=outputs_exp.encoder_hidden_states,
                encoder_attentions=outputs_exp.encoder_attentions,
            )


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return my_shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past




class MultimodalBartClassification(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inned_dim: int,
        num_classes: int,
        pooler_dropout: float
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inned_dim)
        self.dropout = nn.Dropout(p = pooler_dropout)
        self.out_proj = nn.Linear(inned_dim, num_classes)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class MLP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class MyModel(nn.Module):
    def __init__(self, tokenizer, num_classes=2, feature_size=768, max_len=50, explainable=True):
        super(MyModel, self).__init__()
        self.explainable = explainable
        self.bart = MultimodalBartForCausalAnswer.from_pretrained('bart-base')
        self.tokenizer = tokenizer
        
        self.exp_feature_gauss = GaussianEncoder(768, 768)

        self.exp_feature_encoder = BiTransformer(nb_vocab=self.tokenizer.vocab_size, d_model=feature_size, num_head=4)


        
        self.exp_feature_out = MultimodalBartClassification(
            feature_size *2,
            feature_size *2,
            2,
            0
        )
        self.bart._init_weights(self.exp_feature_out.dense)
        self.bart._init_weights(self.exp_feature_out.out_proj)
        self.bart._init_weights(self.exp_feature_encoder.exp_embed)


    

    def forward(self, input_ids, attention_mask, audio_feature, face_feature, posture_feature, video_feature, exp=None, labels=None):
        video_feature = video_feature.float()
        audio_feature = audio_feature.unsqueeze(1)

        visual_feature = torch.concat([video_feature,face_feature,posture_feature],dim=1)
        self.bart.resize_token_embeddings(len(self.tokenizer))
        

        if exp is not None and labels is not None:
            exp_out = self.bart(input_ids=input_ids,
                        attention_mask=attention_mask,
                        acoustic_input=audio_feature,
                        visual_input=visual_feature,
                        exp=exp,
                        labels=labels
                )
            hidden_states = exp_out['encoder_last_hidden_state']
            cls_feat = hidden_states.mean(dim=1)


            exp_loss = exp_out['loss']
            exp_logits = exp_out['logits'].argmax(-1)

            exp_pred_feature = self.exp_feature_encoder(exp_logits).mean(dim=1)

            exp = [[(l if l != -100 else self.tokenizer.pad_token_id) for l in exp_0] for exp_0 in exp]
            exp = torch.tensor([l for l in exp], dtype=torch.long, device=visual_feature.device)
            exp_true_feature = self.exp_feature_encoder(exp).mean(dim=1)

            p_mean, p_var, _ = self.exp_feature_gauss(exp_true_feature)
            q_mean, q_var, q_val = self.exp_feature_gauss(exp_pred_feature)
            ans_feat = torch.concat((cls_feat, q_val), -1)
            output_ans_final = self.exp_feature_out(ans_feat)



            return output_ans_final, exp_loss, q_mean, q_var, p_mean, p_var, exp_logits
        else :
            exp_out = self.bart(input_ids=input_ids,
                        attention_mask=attention_mask,
                        acoustic_input=audio_feature,
                        visual_input=visual_feature,
                        exp=exp,
                        labels=labels
                )
            hidden_states = exp_out['encoder_last_hidden_state']
            cls_feat = hidden_states.mean(dim=1)



            exp_logits = self.bart.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        acoustic_input=audio_feature,
                        visual_input=visual_feature,
                        exp=exp,
                        labels=labels
                        )
            

            exp_pred_feature = self.exp_feature_encoder(exp_logits).mean(dim=1)
            
            q_mean, q_var, q_val = self.exp_feature_gauss(exp_pred_feature)
            ans_feat = torch.concat((cls_feat, q_val), -1)
            output_ans_final = self.exp_feature_out(ans_feat)
            exp_loss,p_mean, p_var = None,None,None

            return output_ans_final, exp_loss, q_mean, q_var, p_mean, p_var, exp_logits
