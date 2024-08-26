# export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/openvla/scripts/extern/load_record_mem.py \
  --model.type "llama2-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "finetune" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs/gemma2newcode" \
  --pretrained_checkpoint "/mnt/csp/mmvision/home/lwh/openvla_runs/jetmoe_llava+llama2-dinosiglip-224px+stage-finetune+x7/checkpoints/step-004255-epoch-00-loss=1.0530.pt" \
  --enable_mixed_precision_training True \
  --continue_from_checkpoint True \
  --continue_step 0 \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 4 \
  --llm_load_weight True
  # --model.finetune_max_steps 1920
