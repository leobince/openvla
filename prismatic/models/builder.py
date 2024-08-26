import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.models.backbones.vision import DinoSigLIPViTBackbone

from scripts.pretrain import PretrainConfig
def load_pretrained_model(cfg: PretrainConfig):
                
        vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )
    
        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=None, inference_mode=True
    )
        vlm = get_vlm(
        cfg.model.model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        continue_from_checkpoint=cfg.continue_from_checkpoint,
    )

        vlm.freeze_backbones(cfg.stage, train_llm_when_align=False)


        vlm.load_from_checkpoint(cfg.stage, run_dir=None, pretrained_checkpoint=cfg.pretrained_checkpoint)
                 
        context_len = cfg.model.llm_max_length


        return tokenizer, vlm, image_transform, context_len
