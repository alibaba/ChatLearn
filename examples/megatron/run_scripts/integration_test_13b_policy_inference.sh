source run_scripts/base_env.sh

cd ${Megatron}/megatron_rlhf
#python train_rlhf.py -c configs_a_dsw_test/rlhf.yaml --exp_name ${exp_name} --change_base_paths base.yaml base.yaml --change_parameter_names loss_on_prompts fix_kl_coef --change_parameter_values True False  # | tee -a ${Megatron}/logs/${exp_name}/log.txt
export exp_name=sft_inf_v1
export tensorboard_dir=/mnt/user/E-tianhang.zhu-364849/cp_tensorboard/${exp_name}
export log_dir=${Megatron}/logs/${exp_name}
export data_checkpoint_path=/mnt/user/E-tianhang.zhu-364849/rlhf_framework/Megatron-v3/data_checkpoint/${exp_name}
export eval_output_dir=${Megatron}/logs/
export eval_data_path=/mnt/shared/Group-m6/tianhang_zhu/move/all_rlhf_data/prompt_rlhf_v3_20w_train_split_08_42.json
#export eval_data_path=/mnt/shared/Group-m6/tianhang_zhu/chatgpt_api_v2/chatgpt_api_v2/rm_eval_repo/eval_dataset/to_collect/online_data_mix_v1_equal_distance_1000_to_collect_sft_chatgpt_responses.jsonl

mkdir ${Megatron}/logs/${exp_name}

build_path=build \
  generation_batch_size=16 \
  old_policy_num_device=8 \
  eval_data_path=${eval_data_path} \
  eval_output_dir=${eval_output_dir} \
  python tests/test_policy_generation.py -c run_configs/configs_13b_policy_inference_v2/rlhf.yaml | tee -a ${Megatron}/logs/${exp_name}/log.txt
