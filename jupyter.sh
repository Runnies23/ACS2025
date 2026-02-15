#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_partition
#SBATCH --output=jupyter.out

if [ $(id -gn) != "render" ]; then
    echo "Switching to group 'render' and re-executing..."
    exec sg render -c "$0 $@"
fi

# --- conda init (สำคัญ!) ---
source /mnt/ASC1664/miniconda3/etc/profile.d/conda.sh
conda activate unifolm-wma

# --- env vars ---
export ROCM_HOME=/opt/rocm-7.1.1
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/mnt/ASC1664/unifolm-wma-0-dual/ASC26-Embodied-World-Model-Optimization/src

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export XFORMERS_FORCE_DISABLE_TRITON=1

python -c "import torch; print(torch.cuda.is_available())"

# --- start jupyter ---
jupyter notebook \
  --no-browser \
  --ip=0.0.0.0 \
  --port=8888
