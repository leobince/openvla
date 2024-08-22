from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import transformers
from torch import nn
from PIL import Image
from PIL import Image
from torchvision import transforms
from typing import Dict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

from PIL import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel
from transformers.models.idefics2.configuration_idefics2 import Idefics2Config, Idefics2VisionConfig
from transformers.models.idefics2.modeling_idefics2 import Idefics2PerceiverResampler, Idefics2VisionEmbeddings, Idefics2Connector
from transformers import SiglipVisionModel, Dinov2Model

from prismatic.models.backbones.vision.base_vision import ImageTransform, LetterboxPad, VisionBackbone, unpack_tuple


class NRPR_DinoSigLIPImageTransform:

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        transform = transforms.Compose([
            transforms.ToTensor(),  
            # 你可以在这里添加更多的变换，例如缩放、裁剪等
        ])

        tensor_img = transform(img)

        return {'image': tensor_img}
    
    
class NRPR_SigLIPViT(nn.Module):
    def __init__(self, config: Idefics2VisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.dtype = torch.float32  # Set a default data type for the model parameters
        self.config = config
        self.embeddings = Idefics2VisionEmbeddings(config)
        siglip_model = SiglipVisionModel.from_pretrained("/mnt/nas/share/home/qbc/ckpt/siglip-base-patch16-224")
        siglip_model = siglip_model.to(self.dtype)
        self.encoder = siglip_model.vision_model.encoder
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.config.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        elif not self._use_flash_attention_2:
            patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class NRPR_DinoViT(nn.Module):
    def __init__(self, config: Idefics2VisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.dtype = torch.float32  # Set a default data type for the model parameters
        self.config = config
        self.embeddings = Idefics2VisionEmbeddings(config)
        dino_model = Dinov2Model.from_pretrained("/mnt/nas/share/home/qbc/ckpt/dinov2-base")
        dino_model = dino_model.to(self.dtype)
        self.encoder = dino_model.encoder
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.config.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        elif not self._use_flash_attention_2:
            patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            hidden_states,
            head_mask=patch_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# Copied from transformers.models

class NRPR_DinoSigLIPViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.config =  Idefics2Config()

        self.sig_vision_model = NRPR_SigLIPViT(self.config.vision_config)
        self.dino_vision_model = NRPR_DinoViT(self.config.vision_config)  # Initialize DinoViT model as well
        self.connector = Idefics2Connector(self.config)
        self.pixel_attention_mask = None
        
        self.dtype = torch.float32  # Set a default data type for the model parameters
        
        self.linear = nn.Linear(4096, 2048)

        self.image_transform = NRPR_DinoSigLIPImageTransform()
        
        
    def forward(
        self,
        pixel_values: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pixel_values = {k: v.to(self.dtype) for k, v in pixel_values.items()}
        pixel_values = pixel_values.get('image', None)
        if pixel_values is None:
            raise("key 'image' not found!!")
        
        image_hidden_states = None
        
        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            
            #batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
            #pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            pixel_values = pixel_values[real_images_inds].contiguous()
            pixel_attention_mask = self.pixel_attention_mask
            
            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            # else:
            #     # Remove padding images from the mask
            #     pixel_attention_mask = pixel_attention_mask.view(
            #         batch_size * num_images, *pixel_attention_mask.shape[2:]
            #     )
            #     pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # Get sequence from the vision encoder
            sig_image_hidden_states = self.sig_vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state
            
            # Modality projection & resampling
            sig_image_hidden_states = self.connector(
                sig_image_hidden_states, attention_mask=patch_attention_mask.view(pixel_values.size(0), -1)
            )
            
            dino_image_hidden_states = self.dino_vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state
            print("dino_image_hidden_states = ", dino_image_hidden_states.shape)

            # Modality projection & resampling
            dino_image_hidden_states = self.connector(
                dino_image_hidden_states, attention_mask=patch_attention_mask.view(pixel_values.size(0), -1)
            )
            
            image_hidden_states = torch.cat([dino_image_hidden_states, sig_image_hidden_states], dim=1)

            # Ensure the unified dtype for image_hidden_states
            image_hidden_states = image_hidden_states.to(dtype=self.dtype)

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=self.device)
            
        # assert image_hidden_states.shape[-1] == 4096
        # image_hidden_states = image_hidden_states.view(-1, 4096)  # -1 会自动计算为 128
        # image_hidden_states = self.linear(image_hidden_states)
        # image_hidden_states = image_hidden_states.view(1, 128, 2048)
        
        return image_hidden_states

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (224, 224, 3)

    @property
    def embed_dim(self) -> int:
        return 4096

    @property
    def num_patches(self) -> int:
        assert self.dino_featurizer.patch_embed.num_patches == self.siglip_featurizer.patch_embed.num_patches
        return self.dino_featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.float32