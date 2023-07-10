ray stop
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=18
# export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=eth0
export RANK=${1:-0}
export LOCAL_MASTER_ADDR=localhost
export CUDA_DEVICE_MAX_CONNECTIONS=1

ports="9000"
for i in $(seq 9001 9050); do
  ports="${ports};${i}"
done
export CUSTOM_PORTS=$ports

rlhf=/mnt/shared/Group-m6/tianhang_zhu/latest_rlhf_0612/rlhf
Megatron=/mnt/shared/Group-m6/tianhang_zhu/latest_rlhf_0612/QWen

rm core*
rm ${Megatron}/megatron/fused_kernels/build/lock

export PYTHONPATH=${Megatron}/megatron_rlhf:${rlhf}:${Megatron}:${PYTHONPATH}

cd ${Megatron}/megatron_rlhf
#python train_rlhf.py -c configs_a_dsw_test/rlhf.yaml --exp_name ${exp_name} --change_base_paths base.yaml base.yaml --change_parameter_names loss_on_prompts fix_kl_coef --change_parameter_values True False  # | tee -a ${Megatron}/logs/${exp_name}/log.txt
export exp_name=eval_test
export tensorboard_dir=/mnt/user/E-tianhang.zhu-364849/cp_tensorboard/${exp_name}
export save=${Megatron}/saved_models
export log_dir=${Megatron}/logs/${exp_name}
export data_checkpoint_path=/mnt/user/E-tianhang.zhu-364849/rlhf_framework/Megatron-v3/data_checkpoint/${exp_name}
export save_episode_interval=2
export eval_episode_interval=2
export eval_output_dir=${Megatron}/eval_output_dir

mkdir ${Megatron}/logs/${exp_name}
# export CUDA_VISIBLE_DEVICES=0
export continue_train=0
export continue_train_global_batch_size=8
export continue_inference_instances=1
export continue_inference_batch_size=4

build_path=buildddd \
  eval_data_num_limit=8 \
  do_math_eval=1 \
  ngram_coef=0 \
  lm_coef=0 \
  math_coef=0 \
  raw_reward_coeff=1 \
  data_path=/mnt/shared/Group-m6/tianhang_zhu/chatgpt_api_v2/chatgpt_api_v2/all_rlhf_data/train_plus_test_use.jsonl \
  eval_data_path=/mnt/shared/Group-m6/tianhang_zhu/chatgpt_api_v2/chatgpt_api_v2/all_rlhf_data/test_use.jsonl \
  python train_rlhf.py -c run_configs/configs_1d3b_4_12/rlhf.yaml 2>&1 | tee -a ${Megatron}/logs/${exp_name}/log.txt

# python eval_pipeline.py -c configs_a_dsw_test_eval/rlhf.yaml

# python tests/test_policy_generation.py -c configs_a_dsw_test/rlhf.yaml

# python tests/get_bias_reward.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_policy_inference.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reward.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reward_forward.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_value_train.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_policy_train.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reference.py -c configs_baseline_seed_100/rlhf.yaml
# python tests/test_reference.py -c configs_baseline_seed_100/rlhf.yaml
