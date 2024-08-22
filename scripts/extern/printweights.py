
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

from transformers import GemmaForCausalLM, AutoConfig, GemmaConfig

# AutoConfig.register("gemma", GemmaConfig)
# gemc = AutoConfig.from_pretrained("/mnt/csp/mmvision/home/lwh/gemma2_2b/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf")
# a = GemmaForCausalLM.from_pretrained("/mnt/csp/mmvision/home/lwh/gemma2_2b/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf")
# for name, param in a.named_parameters():
    # if param.requires_grad:
        # print(f"Parameter name: {name}, Shape: {param.shape}")
        


from prismatic.models.backbones.gemmamoe_project import GemmaMoEForCausalLM, GemmaMoEConfig

AutoConfig.register("gemmamoe", GemmaMoEConfig)
llm_config = AutoConfig.from_pretrained("/mnt/csp/mmvision/home/lwh/gemmamoe")

moe = GemmaMoEForCausalLM._from_config(llm_config)  

        
# moe = GemmaMoEForCausalLM.from_pretrained("/mnt/csp/mmvision/home/lwh/gemmamoe")
for name, param in moe.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}, Shape: {param.shape}")