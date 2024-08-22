import timm

# dino_featurizer = timm.create_model(
            # dino_timm_path_or_url, pretrained=False, num_classes=0, img_size=default_image_size
        # )

import timm
import torch

# 指定模型名称和权重下载路径
model_name = 'vit_so400m_patch14_siglip_224'
weight_path = '/apdcephfs/csp/mmvision/home/lwh/vit/vit_patch14_siglip_224.pth'

# 下载权重
model = timm.create_model("vit_so400m_patch14_siglip_224", pretrained=True, img_size=224)
torch.save(model.state_dict(), weight_path)