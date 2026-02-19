#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_partition
#SBATCH --output=/mnt/%u/run_workspace/%j_profiling.out

if [ $(id -gn) != "render" ]; then
    echo "Switching to group 'render' and re-executing..."
    exec sg render -c "$0 $@"
fi

export ROCM_HOME=/opt/rocm-7.1.1
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH="/mnt/ASC1664/run_workspace/unifolm-world-model-action/src:$PYTHONPATH"

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export XFORMERS_FORCE_DISABLE_TRITON=1

/mnt/ASC1664/miniconda3/envs/unifolm-wma/bin/python -c "import torch; print(torch.cuda.is_available())"

PYTHON=/mnt/ASC1664/miniconda3/envs/unifolm-wma/bin/python
# ckpt=/mnt/ASC1664/unifolm-wma-0-dual_run_workspace/checkpoints/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
# ckpt=/mnt/ASC1664/unifolm-wma-0-dual/ASC26-Embodied-World-Model-Optimization/ckpts/unifolm_wma_dual.ckpt #run work_space weight
ckpt=/mnt/ASC1664/unifolm-wma-0-dual/ASC26-Embodied-World-Model-Optimization/ckpts/unifolm_wma_dual.ckpt
config=./unifolm-world-model-action/configs/inference/world_model_interaction.yaml
res_dir="/mnt/$USER/run_workspace/sh_output/results/unitree_z1_stackbox"
seed=123
model_name=testing

echo "Checking GPU Availability..."
$PYTHON -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'Is ROCm/CUDA available?: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

mkdir -p $res_dir

datasets=("unitree_z1_stackbox")

cd /mnt/ASC1664/run_workspace/

$PYTHON -m xformers.info