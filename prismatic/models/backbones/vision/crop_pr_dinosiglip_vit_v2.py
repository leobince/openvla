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
from torchvision.transforms.functional import resized_crop
from prismatic.models.backbones.vision.base_vision import ImageTransform, LetterboxPad, VisionBackbone, unpack_tuple

import torch
from torchvision import transforms

class CROP_PR_DinoSigLIPImageTransform:
    def __init__(self, target_size):
        self.target_size = target_size

    def crop_image_transform(self, image_tensor):
        # Step 1: Resize to (2*self.target_size, 2*self.target_size)
        scale_height = 2 * self.target_size
        scale_width = 2 * self.target_size
        resize = transforms.Resize((scale_height, scale_width))
        big_image_tensor = resize(image_tensor)

        # Step 2: Split the big image tensor into four parts
        # Top-left
        top_left_tensor = transforms.functional.resized_crop(big_image_tensor, 0, 0, scale_height // 2, scale_width // 2, (self.target_size, self.target_size))
        # Top-right
        top_right_tensor = transforms.functional.resized_crop(big_image_tensor, 0, scale_width // 2, scale_height // 2, scale_width // 2, (self.target_size, self.target_size))
        # Bottom-left
        bottom_left_tensor = transforms.functional.resized_crop(big_image_tensor, scale_height // 2, 0, scale_height // 2, scale_width // 2, (self.target_size, self.target_size))
        # Bottom-right
        bottom_right_tensor = transforms.functional.resized_crop(big_image_tensor, scale_height // 2, scale_width // 2, scale_height // 2, scale_width // 2, (self.target_size, self.target_size))

        resize_small = transforms.Resize((self.target_size, self.target_size))
        small_image_tensor = resize_small(image_tensor)
        
        # Step 3: Concatenate tensors
        concatenated_tensors = torch.cat((top_left_tensor, top_right_tensor, bottom_left_tensor, bottom_right_tensor, small_image_tensor), dim=0)

        return concatenated_tensors

    def __call__(self, img, **kwargs):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Additional transformations can be added here, such as scaling, cropping, etc.
        ])
        tensor_img = transform(img)
        cropped_tensors = self.crop_image_transform(tensor_img)
        print("cropped_tensors = ", cropped_tensors.shape)
        return {'images': cropped_tensors}
    
    
class CROP_PR_SigLIPViT(nn.Module):
    def __init__(self, config: Idefics2VisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.dtype = torch.float32  # Set a default data type for the model parameters
        self.config = config
        siglip_model = SiglipVisionModel.from_pretrained("/mnt/nas/share/home/qbc/ckpt/siglip-base-patch16-224")
        siglip_model = siglip_model.to(self.dtype)
        self.embeddings = siglip_model.vision_model.embeddings
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
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state) if self.use_head else None
        if not return_dict:
            return (last_hidden_state, pooler_output) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CROP_PR_DinoViT(nn.Module):
    def __init__(self, config: Idefics2VisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.dtype = torch.float32  # Set a default data type for the model parameters
        self.config = config
        dino_model = Dinov2Model.from_pretrained("/mnt/nas/share/home/qbc/ckpt/dinov2-base")
        dino_model = dino_model.to(self.dtype)
        self.embeddings = dino_model.embeddings
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
        bool_masked_pos: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
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
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# Copied from transformers.models

class CROP_PR_DinoSigLIP_V2_ViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.config =  Idefics2Config()
        self.default_image_size = default_image_size
        self.sig_vision_model = SiglipVisionModel.from_pretrained("/mnt/nas/share/home/qbc/ckpt/siglip-base-patch16-224")
        self.dino_vision_model = Dinov2Model.from_pretrained("/mnt/nas/share/home/qbc/ckpt/dinov2-base")  # Initialize DinoViT model as well
        self.connector = Idefics2Connector(self.config)
        self.pixel_attention_mask = None
        
        self.dtype = torch.float32  # Set a default data type for the model parameters
        
        self.linear = nn.Linear(4096, 2048)

        self.image_transform = CROP_PR_DinoSigLIPImageTransform(self.default_image_size)
        
        
        
    def forward(
        self,
        pixel_values: Dict[str, torch.Tensor]
    ) -> torch.Tensor:

        pixel_values = {k: v.to(self.dtype) for k, v in pixel_values.items()}
        pixel_values = pixel_values.get('images', None)
        if pixel_values is None:
            raise("key 'images' not found!!")
        
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

            all_hidden_states = []
            for i in range(0, pixel_values.size(1), 3):
                # 提取当前图像的tensor
                current_image_tensor = pixel_values[:, i:i+3, :, :]

                # 通过sig模型处理图像
                sig_tensor_hidden_states = self.sig_vision_model(
                    pixel_values=current_image_tensor,
                ).last_hidden_state

                # 通过dino模型处理图像
                dino_tensor_hidden_states = self.dino_vision_model(
                    pixel_values=current_image_tensor,
                ).last_hidden_state

                # 将两个模型的输出在dim1进行拼接
                combined_hidden_states = torch.cat([dino_tensor_hidden_states, sig_tensor_hidden_states], dim=1)

                print("dino_tensor_hidden_states = ", dino_tensor_hidden_states.shape)
                print("sig_tensor_hidden_states = ", sig_tensor_hidden_states.shape)
                # 将这个图像的特征向量添加到列表中
                all_hidden_states.append(combined_hidden_states)

            # 将所有图像的特征向量在dim0进行拼接
            image_hidden_states = torch.cat(all_hidden_states, dim=0)
            #image_hidden_states.shape =  torch.Size([5, 453, 768])
            # Ensure the unified dtype for image_hidden_states
            image_hidden_states = image_hidden_states.to(dtype=self.dtype)

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=self.device)
            
        # assert image_hidden_states.shape[-1] == 4096
        # image_hidden_states = image_hidden_states.view(-1, 4096)  # -1 会自动计算为 128
        # image_hidden_states = self.linear(image_hidden_states)
        # image_hidden_states = image_hidden_states.view(1, 128, 2048)
        print("image_hidden_states = ", image_hidden_states.shape)
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