#!/usr/bin/sh
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTHONPATH=/HOME/scw6cs5/run/lwh/openvla:$PYTHONPATH
torchrun --standalone --nnodes 1 --nproc_per_node 8 /HOME/scw6cs5/run/lwh/openvla/vla-scripts/train.py \
    --vla.type "prism-dinosiglip-224px+mx-bridge-try" \
    --data_root_dir /HOME/scw6cs5/run/qbc/dataset/OPEN_X_dataset \
    --run_root_dir /HOME/scw6cs5/run/lwh/openvla/runs \
    --prismatic_checkpoint /HOME/scw6cs5/run/qbc/openvla-7b/prism-dinosiglip-224px+7b/ \
    --is_resume False \
    --use_jetmoe True \
    
    
# sbatch --gpus=8 -p vip_gpu_4090_scw6ci6 --output=/HOME/scw6cs5/run/lwh/log/train_output.log --error=/HOME/scw6cs5/run/lwh/log/train_error.log /HOME/scw6cs5/run/lwh/openvla/vla-scripts/train.sh