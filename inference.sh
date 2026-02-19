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

/mnt/ASC1664/miniconda3/envs/unifolm-wma_run/bin/python -c "import torch; print(torch.cuda.is_available())"

PYTHON=/mnt/ASC1664/miniconda3/envs/unifolm-wma_run/bin/python #conda env
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

echo "-------------------------------------------"
echo "Processing dataset: ${dataset}"

ulimit -s unlimited
export XFORMERS_FORCE_DISABLE_TRITON=1

$PYTHON unifolm-world-model-action/scripts/evaluation/world_model_interaction.py \
--seed 123 \
--ckpt_path $ckpt   \
--config $config \
--savedir "/mnt/$USER/run_workspace/result/force/testing/unitree_z1_stackbox" \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 1.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir "./ASC26-Embodied-World-Model-Optimization/unitree_z1_stackbox/case1/world_model_interaction_prompts" \
--dataset "unitree_z1_stackbox" \
--video_length 16 \
--frame_stride 4 \
--n_action_steps 16 \
--exe_steps 16 \
--n_iter 12 \
--timestep_spacing 'uniform_trailing' \
--guidance_rescale 0.7 \
--perframe_ae

# done

echo "All jobs completed!"