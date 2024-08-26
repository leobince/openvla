# export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 1 /mnt/csp/mmvision/home/lwh/openvla/scripts/pretrain.py \
  --model.type "gemma2-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "finetune" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs/gemma2newcode" \
  --pretrained_checkpoint "/mnt/csp/mmvision/home/lwh/openvla_runs/gemma2newcode/llava_more+gemma2-crop-dinosiglip-384px+stage-finetune+x7/checkpoints/step-001500-epoch-00-loss=0.2777.pt" \
  --enable_mixed_precision_training True \
  --continue_from_checkpoint False \
  --continue_step 1930 \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 128 \
  --llm_load_weight False
  # --model.finetune_max_steps 1920
