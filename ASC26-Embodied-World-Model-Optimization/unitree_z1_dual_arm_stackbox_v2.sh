#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_partition
#SBATCH --output=/mnt/ASC1664/unifolm-wma-0-dual/ASC26-Embodied-World-Model-Optimization/%j_unitree_z1_dual_arm_stackbox_v2.out

if [ $(id -gn) != "render" ]; then
    echo "Switching to group 'render' and re-executing..."
    exec sg render -c "$0 $@"
fi

export ROCM_HOME=/opt/rocm-7.1.1
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0

export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export XFORMERS_FORCE_DISABLE_TRITON=1

source /mnt/ASC1664/miniconda3/etc/profile.d/conda.sh
conda activate unifolm-wma

export PYTHONPATH="/mnt/ASC1664/unifolm-wma-0-dual/ASC26-Embodied-World-Model-Optimization/src:$PYTHONPATH"

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

cd /mnt/ASC1664/unifolm-wma-0-dual/ASC26-Embodied-World-Model-Optimization

BASE_DIR="/mnt/ASC1664/unifolm-wma-0-dual/ASC26-Embodied-World-Model-Optimization/unitree_z1_dual_arm_stackbox_v2"

for CASE in case1 case2 case3 case4; do
    echo "========================================"
    echo "Running $CASE"
    echo "========================================"

    bash "$BASE_DIR/$CASE/run_world_model_interaction.sh"

    if [ $? -ne 0 ]; then
        echo "Error occurred in $CASE, stopping."
        exit 1
    fi
done