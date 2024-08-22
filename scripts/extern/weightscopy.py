import torch
    
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

from transformers import GemmaForCausalLM, AutoConfig, GemmaConfig
import os
import sys
sys.path.append('/mnt/csp/mmvision/home/lwh/openvla/')

# 对于vlm的pth paramdict,需要去掉name前的"llm.""
def move_weights_to_moe(moe, paramdict):
    for name, param in moe.named_parameters():
        if param.requires_grad:
            if name in paramdict:
                param = paramdict[name]
                print(name)
            else:
                namesplit = name.split('.')
                if namesplit[-2] == "input_linear" and namesplit[-3] == "experts":
                    namesplit[-3] = 'self_attn'
                    namesplit[-2] = 'q_proj'
                    namesplit.pop(-4)
                    newname = '.'.join(namesplit)
                    newpara = paramdict[newname].unsqueeze(0)
                    newpara = newpara.repeat(8, 1, 1)
                    param = newpara
                elif namesplit[-2] == "output_linear" and namesplit[-3] == "experts":
                    namesplit[-3] = 'self_attn'
                    namesplit[-2] = 'o_proj'
                    namesplit.pop(-4)
                    newname = '.'.join(namesplit)
                    newpara = paramdict[newname].unsqueeze(0)
                    newpara = newpara.repeat(8, 1, 1)
                    param = newpara
                elif namesplit[-2] == "output_linear" and namesplit[-3] == "mlp":
            
                    namesplit[-2] = 'down_proj'
                    newname = '.'.join(namesplit)
                    newpara = paramdict[newname].unsqueeze(0)
                    newpara = newpara.repeat(8, 1, 1)
                    param = newpara
                elif namesplit[-2] == "input_linear" and namesplit[-3] == "mlp":
                    name1 = namesplit
                    name2 = namesplit
                    name1[-2] = "gate_proj"
                    name2[-2] = "up_proj"
                    name1 = '.'.join(name1)
                    name2 = '.'.join(name2)
                    param1 = paramdict[name1]
                    param2 = paramdict[name2]
                    param = torch.cat([param1, param2], dim=0)
                    param = param.unsqueeze(0)
                    param = param.repeat(8, 1, 1)
                else:
                    print(f"Error: {name}")
                print(f"{name} convert")
    return moe


def copy_weights_from_llm():
    # AutoConfig.register("gemma", GemmaConfig)
    # gemc = AutoConfig.from_pretrained("/mnt/csp/mmvision/home/lwh/gemma2_2b/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf")
    gemma = GemmaForCausalLM.from_pretrained("/mnt/csp/mmvision/home/lwh/gemma2_2b/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf")
    paramdict = dict()
    for name, param in gemma.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Shape: {param.shape}")
            paramdict[name] = param
    
    from prismatic.models.backbones.gemmamoe_project import GemmaMoEForCausalLM, GemmaMoEConfig
    AutoConfig.register("gemmamoe", GemmaMoEConfig)
    llm_config = AutoConfig.from_pretrained("/mnt/csp/mmvision/home/lwh/openvla/prismatic/models/backbones/gemmamoe_project/config_jetmoe_to_gemma.json")

    moe = GemmaMoEForCausalLM._from_config(llm_config)                    
    moe = move_weights_to_moe(moe, paramdict)
    torch.save(moe.state_dict(), '/mnt/csp/mmvision/home/lwh/gemmamoe/gemmamoe.pth')
    print('success')
 
    
# vlmpth 中的 llm权重的key名称相比较gemma权重key名称多了个前缀 llm.
def copy_weights_from_vlm():
    pretrained_checkpoint = "/mnt/csp/mmvision/home/lwh/openvla_runs/llava_more+gemma2-dinosiglip-224px+stage-finetune+x7/checkpoints/step-003500-epoch-00-loss=0.0266.pt"
    weight_name = (pretrained_checkpoint.split('/')[-1])
    save_path = pretrained_checkpoint.split('/')[:-1]
    save_path.append('gemmoe')
    save_path = '/'.join(save_path)
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, "gemmamoe_from" + weight_name)
    model_state_dict = torch.load(pretrained_checkpoint)["model"]
    llm_state_dict = model_state_dict['llm_backbone']
    # 首先创建一个没有权重的gemmamoe模型
    from prismatic.models.backbones.gemmamoe_project import GemmaMoEForCausalLM, GemmaMoEConfig
    AutoConfig.register("gemmamoe", GemmaMoEConfig)
    llm_config = AutoConfig.from_pretrained("/mnt/csp/mmvision/home/lwh/openvla/prismatic/models/backbones/gemmamoe_project/config_jetmoe_to_gemma.json")
    moe = GemmaMoEForCausalLM._from_config(llm_config)
    # llm_config = AutoConfig.from_pretrained("/mnt/csp/mmvision/home/lwh/openvla/prismatic/models/backbones/gemmamoe_project/config_jetmoe_to_gemma.json")
    # moe = GemmaMoEForCausalLM._from_config(llm_config)  
    paramdict = dict()
    for key, value in llm_state_dict.items():
        paramdict[key[4:]] = value
    moe = GemmaMoEForCausalLM._from_config(llm_config)                    
    moe = move_weights_to_moe(moe, paramdict) 
    moe_state_dict = moe.state_dict()
    model_state_dict['llm_backbone'] = moe_state_dict
    torch.save(model_state_dict, save_name)
    print('success')
if __name__ == "__main__":
    copy_weights_from_vlm()