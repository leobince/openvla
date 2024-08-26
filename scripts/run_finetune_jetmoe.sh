# export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/openvla/scripts/pretrain.py \
  --model.type "jetmoe-crop-dinosiglip-384px" \
  --dataset.type "llava_more" \
  --stage "finetune" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs" \
  --pretrained_checkpoint "/mnt/csp/mmvision/home/lwh/openvla_runs/jetmoe_llava+jetmoe-dinosiglip-224px+stage-align+x7/checkpoints/step-002907-epoch-00-loss=2.4163.pt" \
  --enable_mixed_precision_training True \
  --continue_from_checkpoint False \
  --continue_step 0 \
  --model.finetune_per_device_batch_size 3 \
  --model.finetune_global_batch_size 24 \
  --llm_load_weight True \
  # --model.arch_specifier "pr"

