# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/HOME/scw6cs5/run/qbc/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/HOME/scw6cs5/run/qbc/miniconda/etc/profile.d/conda.sh" ]; then
        . "/HOME/scw6cs5/run/qbc/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/HOME/scw6cs5/run/qbc/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export PATH=/data/apps/complier/gcc/12.2.0/bin:$PATH
export LD_LIBRARY_PATH=/data/apps/complier/gcc/12.2.0/lib:/data/apps/complier/gcc/12.2.0/lib64:$LD_LIBRARY_PATH
module load cudnn/8.9.3.28_cuda12.x cuda/12.1 
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy='http://10.196.190.146:7890'
export https_proxy='https://10.196.190.146:7890' 
export http_proxy='http://10.196.190.146:7891'
export https_proxy='https://10.196.190.146:7891'

