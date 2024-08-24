#!/usr/bin/sh
export PYTHONPATH=/HOME/scw6cs5/run/lwh/openvla:$PYTHONPATH
python /HOME/scw6cs5/run/lwh/openvla/vla-scripts/deploy.py
# sbatch --gpus=1 -p vip_gpu_4090_scw6ci6 --output=/HOME/scw6cs5/run/lwh/log/deploy_output.log --error=/HOME/scw6cs5/run/lwh/log/deploy_error.log /HOME/scw6cs5/run/lwh/openvla/vla-scripts/deploy.sh