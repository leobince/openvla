export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTHONPATH=/mnt/nas/share2/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/nas/share2/home/lwh/openvla/vla-scripts/train.py \
    --vla.type "prism-dinosiglip-224px+mx-bridge" \
    --data_root_dir /mnt/nas/share/home/qbc/OPEN_X_dataset \
    --run_root_dir /mnt/nas/share2/home/lwh/openvla/runs \
    --prismatic_checkpoint /home/zmz/.cache/huggingface/transformers/openvla-7b/prism-dinosiglip-224px+7b/ \
    --is_resume False
    