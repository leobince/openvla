"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor
from prismatic.models import load_vla


# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.vla = load_vla("/mnt/nas/share/home/qbc/ckpt/prism-dinosiglip-224px+mx-bridge-try+n1+b4+x7/checkpoints/step-027500-epoch-528-loss=0.1609.pt", hf_token=None, load_for_training=False).to(self.device)



    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys() == 1), "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            image = Image.fromarray(image).convert("RGB")
            action = self.vla.predict_action(image = image, instruction = prompt, unnorm_key='calvin_validation')

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "10.71.106.238", port: int = 2022) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = "/home/zmz/.cache/huggingface/transformers/openvla-7b"               # HF Hub Path (or path to local run directory)

    # Server Configuration
    host: str = "10.71.106.238"                                               # Host IP Address
    port: int = 2022                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg.openvla_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
