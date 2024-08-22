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
torchrun \
--nnodes=$WORLD_SIZE \
--node_rank=$RANK \
--master_addr=$MASTER_ADDR \
--nproc_per_node=8 \
--master_port=$MASTER_PORT \ 
/mnt/csp/mmvision/home/lwh/openvla/scripts/pretrain.py \
  --model.type "gemmamoe-dinosiglip-224px" \
  --dataset.type "llava_more" \
  --stage "finetune" \
  --run_root_dir "/mnt/csp/mmvision/home/lwh/openvla_runs/gemmamoe_newcode" \
  --pretrained_checkpoint /mnt/csp/mmvision/home/lwh/openvla_runs/gemma2newcode/llava_more+gemma2-dinosiglip-224px+stage-align+x7/checkpoints/step-002500-epoch-01-loss=2.4788.pt \
  --enable_mixed_precision_training True \
  --continue_from_checkpoint False \
  --continue_step 0 \
  --model.finetune_per_device_batch_size 2 \

