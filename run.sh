#!/bin/bash
#SBATCH --job-name=unifolms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_partition
#SBATCH --output=/mnt/%u/unifolm-wma-0-dual/%j_profiling.out

if [ $(id -gn) != "render" ]; then
    echo "Switching to group 'render' and re-executing..."
    exec sg render -c "$0 $@"
fi

export ROCM_HOME=/opt/rocm-7.1.1
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH="/mnt/ASC1664/unifolm-wma-0-dual/unifolm-world-model-action/src:$PYTHONPATH"

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export XFORMERS_FORCE_DISABLE_TRITON=1

/mnt/ASC1664/miniconda3/envs/unifolm-wma/bin/python -c "import torch; print(torch.cuda.is_available())"

PYTHON=/mnt/ASC1664/miniconda3/envs/unifolm-wma/bin/python
ckpt=/mnt/ASC1664/unifolm-wma-0-dual/checkpoints/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
config=/mnt/ASC1664/unifolm-wma-0-dual/unifolm-world-model-action/configs/inference/world_model_interaction.yaml
res_dir="/mnt/$USER/results"
seed=123
model_name=testing

echo "Checking GPU Availability..."
$PYTHON -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'Is ROCm/CUDA available?: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

mkdir -p $res_dir

datasets=("unitree_z1_stackbox")
n_iters=(12)
fses=(4)

cd /mnt/ASC1664/unifolm-wma-0-dual/unifolm-world-model-action

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    n_iter=${n_iters[$i]}
    fs=${fses[$i]}

    echo "-------------------------------------------"
    echo "Processing dataset: ${dataset}"
    
    $PYTHON scripts/evaluation/world_model_interaction.py \
        --seed ${seed} \
        --ckpt_path $ckpt \
        --config $config \
        --savedir "${res_dir}/${model_name}/${dataset}" \
        --bs 1 --height 320 --width 512 \
        --unconditional_guidance_scale 1.0 \
        --ddim_steps 50 \
        --ddim_eta 1.0 \
        --prompt_dir "examples/world_model_interaction_prompts" \
        --dataset ${dataset} \
        --video_length 16 \
        --frame_stride ${fs} \
        --n_action_steps 16 \
        --exe_steps 16 \
        --n_iter ${n_iter} \
        --timestep_spacing 'uniform_trailing' \
        --guidance_rescale 0.7 \
        --perframe_ae
done

echo "All jobs completed!"