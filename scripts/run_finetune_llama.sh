# export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/openvla/scripts/pretrain.py \
  --model.type "llama2-dinosiglip-224px" \
  --dataset.type "llavanext_onevision" \
  --stage "finetune" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs" \
  --pretrained_checkpoint "/mnt/csp/mmvision/home/lwh/openvla_runs/llava_more+llama2-dinosiglip-224px+stage-finetune+x7/checkpoints/step-006782-epoch-00-loss=0.9244.pt" \
  --enable_mixed_precision_training True \
  --continue_from_checkpoint True \
  --continue_step 0 \
