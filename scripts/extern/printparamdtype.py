from safetensors.torch import load_file
import torch
device2 = torch.device("cuda:2")
device1 = torch.device("cuda:1")
# 指定 safetensors 文件的路径
model_path = "/mnt/csp/mmvision/home/lwh/llama2_7b/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"

# 加载模型文件
state_dict = load_file(model_path, device=device1)

# 查看每个参数的数据类型
for key, tensor in state_dict.items():
    print(f"Parameter: {key}, Data Type: {tensor.dtype}")
    
model_path = "/mnt/csp/mmvision/home/lwh/gemma2_2b/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"

# 加载模型文件
state_dict = load_file(model_path, device=device2)

# 查看每个参数的数据类型
for key, tensor in state_dict.items():
    print(f"Parameter: {key}, Data Type: {tensor.dtype}")