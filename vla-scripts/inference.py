"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`
    - If you want to suppress random Tensorflow logs --> `export TF_CPP_MIN_LOG_LEVEL=3`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""
import numpy as np
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict, Union
import draccus
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.models.vlms import PrismaticVLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoModelForVision2Seq, AutoProcessor
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

def inference(
    payload: Dict[str, Any],
    ) -> None:
    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    hf_token = None
    device = torch.device("cuda:1")
    vla = load_vla("/mnt/nas/share/home/qbc/ckpt/prism-dinosiglip-224px+mx-bridge-try+n1+b4+x7/checkpoints/step-027500-epoch-528-loss=0.1609.pt", hf_token=hf_token, load_for_training=False).to(device)
    openvla_path = "/home/zmz/.cache/huggingface/transformers/openvla-7b"
    image, instruction = payload["image"], payload["instruction"]
    prompt = get_openvla_prompt(instruction, openvla_path)
    image = Image.fromarray(image).convert("RGB")
    action = vla.predict_action(image = image, instruction = prompt, unnorm_key='calvin_validation')
    print(action)
    return action


if __name__ == "__main__":
    step = np.load("/mnt/nas/share/home/qbc/openvla/episode_0553567.npz",allow_pickle=True)
    images = step["rgb_static"]
    payload = {"image": images, "instruction": "do something"}
    inference(payload)
