# Run from the root of the repository
export NCCL_P2P_DISABLE="0"
export NCCL_IB_DISABLE="0"
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/qbc/openvla/scripts/pretrain.py \
  --model.type "jetmoe-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "finetune" \
  --model.align_global_batch_size 40 \
  --model.align_per_device_batch_size 5 \
  --run_id "crop_pr" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/qbc/opevla/runs" \
  --model.vision_backbone_id "crop_dinosiglip-vit" \
  --model.image_resize_strategy "resize-naive" \
  --continue_from_checkpoint False \
  --enable_mixed_precision_training True \
  --continue_step 0 \
  --pretrained_checkpoint "/mnt/csp/mmvision/home/lwh/qbc/moe/runs/crop_pr/checkpoints/step-003000-epoch-01-loss=9.4532.pt" \
  --model.arch_specifier "pr" 

#   --model.model_id "jetmoe-dinosiglip-224px" \
#   --model.vision_backbone_id "nrpr_dinosiglip-vit" \
#   --model.image_resize_strategy "letterbox" \
#   --model.llm_backbone_id "vicuna-v15-7b" 
#   --model.vision_backbone_id "dinosiglip-vit-so-384px" \
#  --model.vision_backbone_id "crop_pr_dinosiglip_v2-vit" \
#  --model.vision_backbone_id "crop_dinosiglip-vit" \
