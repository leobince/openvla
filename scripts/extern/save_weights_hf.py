import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

# 假设你有一个 .pth 文件和相关文件
pth_file = "/mnt/csp/mmvision/home/lwh/gemmamoeold/gemmamoe.pth"


# 加载 .pth 权重文件
weights = torch.load(pth_file)

from prismatic.models.backbones.gemmamoe_project import GemmaMoEForCausalLM, GemmaMoEConfig

AutoConfig.register("gemmamoe", GemmaMoEConfig)
llm_config = AutoConfig.from_pretrained("/mnt/csp/mmvision/home/lwh/openvla/prismatic/models/backbones/gemmamoe_project/config_jetmoe_to_gemma.json")

weights = torch.load(pth_file)
moe = GemmaMoEForCausalLM._from_config(llm_config)  


moe.load_state_dict(weights)

# 加载配置文件和词汇表
tokenizer = AutoTokenizer.from_pretrained("/mnt/csp/mmvision/home/lwh/gemma2_2b/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf")

# 保存模型和 tokenizer 为 HF 格式
output_dir = "/mnt/csp/mmvision/home/lwh/gemmamoe"
moe.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
llm_config.save_pretrained(output_dir)