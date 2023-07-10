ray stop

export NCCL_SOCKET_IFNAME=eth0
export RANK=${1:-0}
export LOCAL_MASTER_ADDR=localhost

ports="9000"
for i in $(seq 9001 9050); do
  ports="${ports};${i}"
done
export CUSTOM_PORTS=$ports

rlhf=/mnt/alinlp/tianhang_workspace/new_rlhf/rlhf
Megatron=/mnt/alinlp/tianhang_workspace/new_rlhf/Megatron-v3

rm core*
rm ${Megatron}/megatron/fused_kernels/build/lock

export PYTHONPATH=${Megatron}/examples/chatgpt/rlhf_demo:${rlhf}:${Megatron}:${PYTHONPATH}

cd ${Megatron}/examples/chatgpt/rlhf_demo
#python train_rlhf.py -c configs_a_dsw_test/rlhf.yaml --exp_name ${exp_name} --change_base_paths base.yaml base.yaml --change_parameter_names loss_on_prompts fix_kl_coef --change_parameter_values True False  # | tee -a ${Megatron}/logs/${exp_name}/log.txt

exp_name="eval_both" \
  build_path=${exp_name} \
  policy_load=/cpfs01/shared/Group-m6/tianhang_zhu/checkpoints/13b.new1.data-v10.15.5-13B-bf16-mp4-pp1-lr-1e-5-minlr-1e-6-bs-128-gpus-64-seqlen-8192 \
  policy_load_iteration=8000 \
  policy_inference_temperature=0.5 \
  python eval_pipeline.py -c configs_13b_eval_shared.yaml | tee -a ${Megatron}/logs/${exp_name}/log.txt

# python train_rlhf.py -c configs_a_dsw_test/rlhf.yaml | tee -a ${Megatron}/logs/${exp_name}/log.txt

# python tests/get_bias_reward.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_policy_generation.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_policy_inference.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reward.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reward_forward.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_value_train.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_policy_train.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reference.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reference.py -c configs_baseline_seed_100/rlhf.yaml
