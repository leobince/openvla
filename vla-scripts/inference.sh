#!/usr/bin/sh
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTHONPATH=/HOME/scw6cs5/run/lwh/openvla:$PYTHONPATH
python /HOME/scw6cs5/run/lwh/openvla/vla-scripts/inference.py 
    
    
# sbatch --gpus=1 -p vip_gpu_4090_scw6ci6 --output=/HOME/scw6cs5/run/lwh/log/inference_output.log --error=/HOME/scw6cs5/run/lwh/log/inference_error.log /HOME/scw6cs5/run/lwh/openvla/vla-scripts/inference.sh