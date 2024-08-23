export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_DISABLE=0
export GLOO_SOCKET_IFNAME=bond1


export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8192
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

