from safetensors import safe_open
import torch
from os import path

model_path = "/home/zmz/.cache/huggingface/transformers/openvla-7b"
# 假设你的三个 safetensors 文件名为 model_part1.safetensors, model_part2.safetensors, model_part3.safetensors
file_paths = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]

# 创建一个空的 state_dict 来存储合并后的权重
merged_state_dict = {}

# 遍历每个文件并读取权重
for file_path in file_paths:
    file_path = path.join(model_path, file_path)
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            print(key)
            merged_state_dict[key] = f.get_tensor(key)

assert (
            "projector" in merged_state_dict 
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector`!"

# 现在 merged_state_dict 包含了所有合并后的权重
# 你可以将其加载到你的模型中
# model = YourModelClass()  # 替换为你的模型类
# model.load_state_dict(merged_state_dict)
# model = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
# 现在模型已经加载了所有的权重