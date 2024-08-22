# export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/openvla/scripts/pretrain.py \
  --model.type "gemma2-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "finetune" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs/gemma2newcode" \
  --pretrained_checkpoint "/mnt/csp/mmvision/home/lwh/openvla_runs/gemma2newcode/llava_more+gemma2-dinosiglip-224px+stage-align+x7/checkpoints/step-002500-epoch-01-loss=2.4788.pt" \
  --enable_mixed_precision_training True \
  --continue_from_checkpoint False \
  --continue_step 0 \
