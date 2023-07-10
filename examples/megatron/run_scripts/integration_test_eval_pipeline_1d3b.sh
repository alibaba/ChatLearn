ray stop

export NCCL_SOCKET_IFNAME=eth0
export RANK=${1:-0}
export LOCAL_MASTER_ADDR=localhost
export CUDA_DEVICE_MAX_CONNECTIONS=1

ports="9000"
for i in $(seq 9001 9050); do
  ports="${ports};${i}"
done
export CUSTOM_PORTS=$ports

rlhf=/mnt/user/E-tianhang.zhu-364849/rlhf_framework_QWen/rlhf
Megatron=/mnt/user/E-tianhang.zhu-364849/rlhf_framework_QWen/QWen

rm core*
rm ${Megatron}/megatron/fused_kernels/build/lock

export PYTHONPATH=${Megatron}/megatron_rlhf:${rlhf}:${Megatron}:${PYTHONPATH}

cd ${Megatron}/megatron_rlhf
#python train_rlhf.py -c configs_a_dsw_test/rlhf.yaml --exp_name ${exp_name} --change_base_paths base.yaml base.yaml --change_parameter_names loss_on_prompts fix_kl_coef --change_parameter_values True False  # | tee -a ${Megatron}/logs/${exp_name}/log.txt
export exp_name=eval_test_13b

# mkdir ${Megatron}/logs/${exp_name}

build_path=${exp_name} \
  generation_batch_size=4 \
  reward_load=/cpfs01/shared/Group-m6/yuanzheng.yz/QWen_rm/reward_model_aligptdata/checkpoint/Reward-Model-refine-m6rm.13b.sft-13B-bf16-mp4-pp1-lr-2e-5-minlr-1e-6-bs-32-gpus-8-seqlen-1024 \
  reward_load_iteration=10000 \
  python eval_pipeline.py -c run_configs/configs_13b_16g_eval/rlhf.yaml | tee -a ${Megatron}/logs/${exp_name}/log.txt

# python tests/test_policy_generation.py -c configs_a_dsw_test/rlhf.yaml

# python tests/get_bias_reward.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_policy_inference.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reward.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reward_forward.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_value_train.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_policy_train.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reference.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reference.py -c configs_baseline_seed_100/rlhf.yaml
