# export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/openvla/scripts/pretrain.py \
  --model.type "phi-2-3b-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "finetune" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs/gemma2newcode" \
  --model.arch_specifier "no-align-fused-gelu-mlp" \
  --enable_mixed_precision_training True \
  --continue_from_checkpoint False \
  --continue_step 0 \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 16 \
  --llm_load_weight False
  # --model.finetune_max_steps 1920
