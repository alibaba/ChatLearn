source base_env.sh

export exp_name=run_test_13b_tp8_meg2

# mkdir ${Megatron}/logs/${exp_name}

# build_path=build \
# generation_batch_size=4 \
# num_device=8 \
# tp=8 \
# eval_data_path="/mnt/shared/Group-m6/tianhang_zhu/move/all_rlhf_data/prompt_rlhf_v3_20w_train_split_08_42.json" \
# policy_load=/cpfs01/shared/Group-m6/xianyan.xianyanjia/llama/megatron_models/vicuna-13b-new-tp8/ \
# eval_output_dir=${Megatron}/logs/$(date +%F) \
# python tests/test_policy_generation.py -c llama/configs_13b_policy_inference/rlhf.yaml #> ${Megatron}/logs/${exp_name}_log.txt 2>&1&

TP=8
export exp_name=run_test_13b_tp${TP}_meg2_meg_rotary

# policy_load=/cpfs01/shared/Group-m6/xianyan.xianyanjia/llama/megatron_models/vicuna-13b-rms/ \
build_path=build \
  generation_batch_size=4 \
  num_device=$TP \
  tp=$TP \
  eval_data_path="/mnt/shared/Group-m6/tianhang_zhu/move/all_rlhf_data/prompt_rlhf_v3_20w_train_split_08_42.json" \
  policy_load=/cpfs01/shared/Group-m6/xianyan.xianyanjia/llama/megatron_models/vicuna-13b-new-tp8/ \
  eval_output_dir=${Megatron}/logs/$(date +%F) \
  python tests/test_policy_generation.py -c llama/configs_13b_policy_inference/rlhf.yaml
