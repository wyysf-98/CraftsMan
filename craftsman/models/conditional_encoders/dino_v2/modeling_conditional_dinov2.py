# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Reference:
# * transformers/models/dinov2/modeling_dinov2.py
# * https://github.com/facebookresearch/DiT/blob/main/models.py#L101
# * https://github.com/3DTopia/OpenLRM/tree/main/openlrm/models/encoders/dinov2
""" PyTorch DINOv2 model."""

from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from .modeling_dinov2 import (
    Dinov2Config,
    Dinov2Layer,
    Dinov2Model,
    Dinov2Embeddings,
    BaseModelOutput,
    BaseModelOutputWithPooling,
)


class ModLN(nn.Module):
    def __init__(self, inner_dim: int, mod_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x:torch.Tensor, condition:torch.Tensor):
        '''
        x: [N, M, C_in], M: num of tokens
        condition: [N, C_mod]
        '''
        shift, scale = self.mlp(condition).unsqueeze(1).chunk(2, dim=-1)
        return x * (1 + scale) + shift


class ConditionalDinov2Config(Dinov2Config):
    def __init__(self, modulation_dim: int = 1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modulation_dim = modulation_dim


class ConditionalDinov2Layer(Dinov2Layer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: ConditionalDinov2Config) -> None:
        super().__init__(config)
        self.mod_norm1 = ModLN(config.hidden_size, config.modulation_dim)
        self.mod_norm2 = ModLN(config.hidden_size, config.modulation_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.mod_norm1(self.norm1(hidden_states), condition),  # in Dinov2, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.mod_norm2(self.norm2(hidden_states), condition)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->Dinov2
class ConditionalDinov2Encoder(nn.Module):
    def __init__(self, config: ConditionalDinov2Config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ConditionalDinov2Layer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        condition: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    condition,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, 
                    layer_head_mask, 
                    condition,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ConditionalDinov2Model(Dinov2Model):
    config_class = ConditionalDinov2Config
    def __init__(self, config: ConditionalDinov2Config):
        super().__init__(config)
        self.config = config

        self.embeddings = Dinov2Embeddings(config)
        self.encoder = ConditionalDinov2Encoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            condition=condition,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

