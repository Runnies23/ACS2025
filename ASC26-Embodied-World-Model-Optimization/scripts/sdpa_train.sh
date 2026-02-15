#!/bin/bash
#SBATCH --job-name=b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --partition=gpu_partition
#SBATCH --output=/mnt/ASC1664/unifolm-wma-0-dual/ASC26-Embodied-World-Model-Optimization/scripts/%j.out

if [ $(id -gn) != "render" ]; then
    echo "Switching to group 'render' and re-executing..."
    exec sg render -c "$0 $@"
fi

export ROCM_HOME=/opt/rocm-7.1.1
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0

export HIP_VISIBLE_DEVICES=0

source /mnt/ASC1664/miniconda3/etc/profile.d/conda.sh
conda activate unifolm-wma

export PYTHONPATH="/mnt/ASC1664/unifolm-wma-0-dual/ASC26-Embodied-World-Model-Optimization/src:$PYTHONPATH"

name="use_SPDA"
config_file="configs/train/config.yaml"
save_root="/mnt/ASC1664/unifolm-wma-0-dual/use_SPDA_weight"

mkdir -p $save_root/$name

cd /mnt/ASC1664/unifolm-wma-0-dual/ASC26-Embodied-World-Model-Optimization

python scripts/trainer.py \
  --base $config_file \
  --train \
  --name $name \
  --logdir $save_root \
  --devices 1 \
  --total_gpus=1 \
  lightning.trainer.num_nodes=1 \
  lightning.trainer.strategy=auto
