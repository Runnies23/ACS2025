model_name=testing
ckpt=unifolm-wma-0-dual_run_workspace/checkpoints/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
config=configs/inference/world_model_interaction.yaml
seed=123
res_dir="unifolm-wma-0-dual_run_workspace/results"

datasets=(
    "unitree_z1_stackbox"
)

#n_iters=(12 7 11 8 11)
#fses=(4 4 4 4 6)
n_iters=(12)
fses=(4)

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    n_iter=${n_iters[$i]}
    fs=${fses[$i]}

    CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/world_model_interaction.py \
    --seed ${seed} \
    --ckpt_path $ckpt \
    --config $config \
    --savedir "${res_dir}/${model_name}/${dataset}" \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 1.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt_dir "unifolm-wma-0-dual/unifolm-world-model-action/examples/world_model_interaction_prompts" \
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
