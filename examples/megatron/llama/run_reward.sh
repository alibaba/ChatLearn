source base_env.sh

export exp_name=run_test_13b_tp8_meg2

# mkdir ${Megatron}/logs/${exp_name}

# build_path=build \
# generation_batch_size=4 \
# eval_data_path="/mnt/shared/Group-m6/tianhang_zhu/move/all_rlhf_data/prompt_rlhf_v3_20w_train_split_08_42.json" \
# reward_load=/cpfs01/shared/Group-m6/xianyan.xianyanjia/llama/megatron_models/models--IDEA-CCNL--Ziya-LLaMA-7B-Reward-ln/ \
# eval_output_dir=${Megatron}/logs/$(date +%F) \
# python tests/test_reward_forward.py -c llama/configs_13b_7b/test_reward.yaml

build_path=build \
  generation_batch_size=4 \
  reward_tp=8 \
  reward_device=8 \
  eval_output_dir=${Megatron}/logs/$(date +%F) \
  reward_load=/cpfs01/shared/Group-m6/yuanhongyi.yhy/megatron_llama/Megatron-LM-new0704/experiments/new_tokenizer_llamasft_hh_rm_2023-07-05_gpt_13B_1w8g_tp8_pp1_mb4_seqlen2048_gbs64_sp_do_bf16/ \
  reward_load_iteration=1000 \
  python tests/test_reward_forward.py -c llama/configs_13b_16g_pipe/test_reward.yaml
