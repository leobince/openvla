"""
llama2.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""

from typing import Optional, Sequence, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    LLaMa2ChatPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
)

from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

from transformers import GemmaForCausalLM, GemmaConfig
# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
Gemma_MODELS = {
    
    "gemma-2b":{
        "llm_family": "gemma", "llm_cls": GemmaForCausalLM, "hf_hub_path": "/mnt/csp/mmvision/home/lwh/gemma-2b/models--google--gemma-2b/snapshots/68e273d91b1d6ea57c9e6024c4f887832f7b43fa"
    },
   
}
# fmt: on


class GemmaLLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
        debug = False,
        llm_load_weight: bool = True
  
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            llm_load_weight=llm_load_weight,
            **Gemma_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        # if self.identifier.startswith("llama2-") and self.identifier.endswith("-pure"):
        return PurePromptBuilder

        # elif self.identifier.startswith("llama2-") and self.identifier.endswith("-chat"):
            # return LLaMa2ChatPromptBuilder

        # elif self.identifier.startswith("vicuna"):
            # return VicunaV15ChatPromptBuilder

        # raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return GemmaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:

        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)
