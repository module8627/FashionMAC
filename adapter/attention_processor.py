from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.lora import LoRALinearLayer
from torchvision.transforms import Resize, InterpolationMode


class SAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            parse_encoder_hidden_states=None,
            pre_step_mask = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SimpleFeatureMaskAdapter(nn.Module):
    def __init__(self, input_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.input_dim = input_dim

        channels = [256, 128, 64]

        self.conv_net = nn.Sequential(
            nn.BatchNorm2d(input_dim),

            nn.Conv2d(input_dim, channels[0], kernel_size=1), 
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channels[2], num_features, kernel_size=1),
            nn.Sigmoid()
        )
        

    def forward(self, x, h, w):
        """x: [batch, seq_len, input_dim]"""
        batch_size, seq_len, dim = x.shape
        
        x_spatial = x.permute(0, 2, 1).view(batch_size, dim, h, w).contiguous()
        mask = self.conv_net(x_spatial)
        mask = mask.view(batch_size, self.num_features, -1).permute(0, 2, 1)
        
        return mask


class CAttnProcessor2_0_RADA(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, h = 0, w = 0, cross_attention_dim=None, context_num=0, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.h = h // 8
        self.w = w * 2 // 8
        self.scale = scale / context_num if context_num > 0 else scale
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.context_num = context_num
        self.mask_loss = None
        self.now_step_pred_mask = None
        self.to_k_layers = nn.ModuleList([
            nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            for _ in range(context_num)
        ])
        
        self.to_v_layers = nn.ModuleList([
            nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            for _ in range(context_num)
        ])
        self.mask_adapter = SimpleFeatureMaskAdapter(
            input_dim=hidden_size,
            num_features=context_num
        )

    def load_state_dict_from_original(self, original_state_dict, layer_name):
        """
        Initialize multi-context layers from original cross attention weights
        """
        mapping = {}
        
        if f"{layer_name}.to_k.weight" in original_state_dict:
            orig_k_weight = original_state_dict[f"{layer_name}.to_k.weight"]
            orig_v_weight = original_state_dict[f"{layer_name}.to_v.weight"]
            
            for i in range(self.context_num):
                mapping[f"to_k_layers.{i}.weight"] = orig_k_weight
                mapping[f"to_v_layers.{i}.weight"] = orig_v_weight
        new_state_dict = {}
        for key, value in mapping.items():
            new_state_dict[key] = value
        self.load_state_dict(new_state_dict, strict=False)
        
        print(f"✓ Initialized CAttnProcessor2_0_RADA for {layer_name} with {self.context_num} contexts")


    def get_mask_loss(self):
        return self.mask_loss
    
    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            parse_encoder_hidden_states=None,
            pre_step_mask = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states_mask_pred_input = hidden_states
        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if parse_encoder_hidden_states is not None:
            adapter_scale = (self.h * self.w / hidden_states.shape[1]) ** 0.5
            adapter_h = int(self.h // adapter_scale)
            adapter_w = int(self.w // adapter_scale)
            if hasattr(self, 'mask_adapter'):
                latents_parse_mask_pred = self.mask_adapter(hidden_states_mask_pred_input, adapter_h, adapter_w)
                latents_parse_mask_pred = latents_parse_mask_pred.permute(2, 0, 1)

                latents_parse_mask = latents_parse_mask_pred
                if self.name.endswith("attn2.processor") and self.name.startswith("up_blocks.3.attentions.2"):
                    self.now_step_pred_mask = latents_parse_mask_pred.detach()
                    self.now_step_pred_mask = self.now_step_pred_mask.view(self.context_num, batch_size, adapter_h, adapter_w)
                else:
                    self.now_step_pred_mask = None
                if pre_step_mask is not None and self.name.startswith('down_blocks'):
                    latents_parse_mask = pre_step_mask
                    latents_parse_mask = Resize([adapter_h, adapter_w],InterpolationMode.NEAREST)(latents_parse_mask)
                    latents_parse_mask = latents_parse_mask.view(self.context_num, batch_size, adapter_h * adapter_w)
                    
            
            for i, context in enumerate(parse_encoder_hidden_states):
                if context is None:
                    num_valid_contexts -= 1
                    continue
                if attn.norm_cross:
                    context = attn.norm_encoder_hidden_states(context)
                context_idx = min(i, self.context_num - 1)
                parse_txt_key = self.to_k_layers[context_idx](context)
                parse_txt_value = self.to_v_layers[context_idx](context)

                parse_txt_key = parse_txt_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                parse_txt_value = parse_txt_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                parse_txt_hidden_states = F.scaled_dot_product_attention(
                    query, parse_txt_key, parse_txt_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                parse_txt_hidden_states = parse_txt_hidden_states.transpose(1, 2).reshape(batch_size, -1,
                                                                                          attn.heads * head_dim)
                parse_txt_hidden_states = parse_txt_hidden_states.to(query.dtype)
                parse_txt_hidden_states = parse_txt_hidden_states * latents_parse_mask[i].unsqueeze(-1)
                hidden_states = hidden_states + parse_txt_hidden_states * self.scale
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states