export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTHONPATH=/HOME/scw6cs5/run/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 2 --nproc-per-node 8 /HOME/scw6cs5/run/lwh/openvla/vla-scripts/train.py \
    --vla.type "prism-dinosiglip-224px+mx-bridge" \
    --data_root_dir /HOME/scw6cs5/run/qbc/dataset/OPEN_X_dataset \
    --run_root_dir /HOME/scw6cs5/run/lwh/openvla/runs \
    --prismatic_checkpoint /HOME/scw6cs5/run/qbc/openvla-7b/prism-dinosiglip-224px+7b/ \
    --is_resume False \
    
    
# sbatch -N 2 -p vip_gpu_4090_scw6ci6  --gres=gpu:8  --qos=gpugpu --output=/HOME/scw6cs5/run/lwh/log/train_output.log --error=/HOME/scw6cs5/run/lwh/log/train_error.log /HOME/scw6cs5/run/lwh/openvla/vla-scripts/train.sh