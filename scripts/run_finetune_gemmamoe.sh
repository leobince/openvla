











#/bin/bash
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/openvla/scripts/pretrain.py \
  --model.type "gemmamoe-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "finetune" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs/gemmamoe_newcode" \
  --pretrained_checkpoint /mnt/csp/mmvision/home/lwh/openvla_runs/gemmamoe_newcode/llava_more+gemmamoe-dinosiglip-224px+stage-finetune+x7/checkpoints/step-001920-epoch-00-loss=1.2278.pt \
  --enable_mixed_precision_training True \
  --continue_from_checkpoint True \
  --continue_step 1930 \
  --model.finetune_per_device_batch_size 1 \
  --llm_load_weight False 
  # --model.finetune_max_steps 1930







