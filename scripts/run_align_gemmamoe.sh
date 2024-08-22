# export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/openvla/scripts/pretrain.py \
  --model.type "gemmamoe-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "align" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs/" \
  --enable_mixed_precision_training True \
  --pretrained_checkpoint /mnt/csp/mmvision/home/lwh/openvla_runs/llava_more+gemmamoe-dinosiglip-224px+stage-align+x7/checkpoints/step-000500-epoch-00-loss=10.5699.pt \
  --continue_from_checkpoint False \
  --continue_step 0 \
  --llm_load_weight False \
  --model.align_per_device_batch_size 4 \
  --model.align_epoch 3 \