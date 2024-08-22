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
from einops import rearrange

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
        return {'images': cropped_tensors}
    
    


class CROP_PR_DinoSigLIP_V3_ViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 384) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.config =  Idefics2Config()
        self.default_image_size = default_image_size
        self.sig_vision_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        self.dino_vision_model = Dinov2Model.from_pretrained("facebook/dinov2-large")  # Initialize DinoViT model as well
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
            
            
            pixel_values = rearrange(pixel_values, 'b (n c) h w -> (b n) c h w', c=3)
            
            sig_tensor_hidden_states = self.sig_vision_model(
                pixel_values=pixel_values,
            ).last_hidden_state
            #384: sig_tensor_hidden_states =  torch.Size([5, 729, 1152])

            sig_tensor_hidden_states = nn.functional.pad(sig_tensor_hidden_states, (0, 0, 0, 1))
            #384: sig_tensor_hidden_states =  torch.Size([5, 730, 1152])

            # 通过dino模型处理图像
            dino_tensor_hidden_states = self.dino_vision_model(
                pixel_values=pixel_values,
            ).last_hidden_state

            #224: sig_tensor_hidden_states.shape =  torch.Size([5, 196, 768])
            #224: dino_tensor_hidden_states.shape =  torch.Size([5, 257, 768])
            #384: sig_tensor_hidden_states =  torch.Size([5, 729, 1152])
            #384: dino_tensor_hidden_states =  torch.Size([5, 730, 1024])

            # 将所有图像的特征向量在dim0进行拼接
            image_hidden_states = torch.cat([sig_tensor_hidden_states, dino_tensor_hidden_states], dim=2)
            #224: image_hidden_states.shape =  torch.Size([5, 453, 768])

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
        return (384, 384, 3)

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