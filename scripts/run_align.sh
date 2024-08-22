# export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/openvla/scripts/pretrain.py \
  --model.type "gemma2-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "align" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs/gemma2newcode" \
  --enable_mixed_precision_training True \
  --model.align_epochs 2 \
  --model.align_per_device_batch_size 16 \


torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/openvla/scripts/pretrain.py \
  --model.type "gemma2-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "align" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs/gemma2newcode/align-llm" \
  --enable_mixed_precision_training True \
  --model.align_epochs 2 \
  --model.align_per_device_batch_size 16 \
  --train_llm_when_align True \

  # --pretrained_checkpoint /mnt/csp/mmvision/home/lwh/openvla_runs/llava_more+gemma2-dinosiglip-224px+stage-align+x7/checkpoints/step-000500-epoch-00-loss=6.6432.pt \
  # --continue_from_checkpoint True \
  # --continue_step 500 \



