export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_IB_TIMEOUT=22
export PYTHONPATH=/mnt/csp/mmvision/home/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc-per-node 8 /mnt/csp/mmvision/home/lwh/openvla/vla-scripts/train.py \
    --vla.type "jetmoe-siglip-224px+mx-bridge" \
    --data_root_dir /mnt/csp/mmvision/home/lwh/qbc/OPEN_X_dataset \
    --run_root_dir /mnt/csp/mmvision/home/lwh/openvla_vlaruns \
    --prismatic_checkpoint /mnt/csp/mmvision/home/lwh/openvla_runs/llava_more+jetmoe-dinosiglip-224px+stage-finetune+x7/checkpoints/step-006782-epoch-00-loss=1.1605.pt \
    --is_resume False
    