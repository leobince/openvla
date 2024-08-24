# Run from the root of the repository
export NCCL_P2P_DISABLE="0"
export NCCL_IB_DISABLE="0"
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/qbc/origin/openvla/scripts/pretrain.py \
  --model.type "jetmoe-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "align" \
  --model.align_global_batch_size 200 \
  --model.align_per_device_batch_size 25 \
  --run_id "crop824" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/qbc/origin/moeorigin/runs" \
  --model.vision_backbone_id "crop_dinosiglip-vit" \
  --model.image_resize_strategy "resize-naive" \
  --model.arch_specifier "pr" \
  --llm_load_weight True


#  --pretrained_checkpoint "/mnt/csp/mmvision/home/lwh/qbc/moe/runs/crop_pr/checkpoints/step-004500-epoch-01-loss=9.3942.pt" \
#  --model.arch_specifier "pr" 

#   --model.model_id "jetmoe-dinosiglip-224px" \
#   --model.vision_backbone_id "nrpr_dinosiglip-vit" \
#   --model.image_resize_strategy "letterbox" \
#   --model.llm_backbone_id "vicuna-v15-7b" 
#   --model.vision_backbone_id "dinosiglip-vit-so-384px" \
#  --model.vision_backbone_id "crop_pr_dinosiglip_v2-vit" \
#  --model.vision_backbone_id "crop_dinosiglip-vit" \
