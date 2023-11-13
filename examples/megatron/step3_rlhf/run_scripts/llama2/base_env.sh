source run_scripts/base_env.sh
[ -z "$num_device_policy" ] && export num_device_policy=$num_device
[ -z "$num_device_ref" ] && export num_device_ref=$num_device
[ -z "$num_device_reward" ] && export num_device_reward=$num_device
[ -z "$num_device_value" ] && export num_device_value=$num_device
[ -z "$num_device_ppo_policy" ] && export num_device_ppo_policy=$num_device
[ -z "$num_device_ppo_value" ] && export num_device_ppo_value=$num_device

export PYTHONPATH=${PYTHONPATH}:${CHATLEARN}/examples/megatron/step2_reward

if [[ "$model_size" == "7B" ]]; then
  export policy_num_layers=32
  export policy_hidden_size=4096
  export policy_num_attention_heads=32
  export policy_num_query_groups=32
  export policy_ffn_hidden_size=11008
  export reward_num_layers=32
  export reward_hidden_size=4096
  export reward_num_query_groups=32
  export reward_num_attention_heads=32
  export reward_ffn_hidden_size=11008
  export max_position_embedding=2048
elif [[ "$model_size" == "13B" ]]; then
  export policy_num_layers=40
  export policy_hidden_size=5120
  export policy_num_attention_heads=40
  export policy_ffn_hidden_size=13824
  export policy_num_query_groups=40
  export reward_num_layers=40
  export reward_hidden_size=5120
  export reward_num_attention_heads=40
  export reward_ffn_hidden_size=13824
  export reward_num_query_groups=40
elif [[ "$model_size" == "70B" ]]; then
  export policy_num_layers=80
  export policy_hidden_size=8192
  export policy_num_attention_heads=64
  export policy_ffn_hidden_size=28672
  export policy_num_query_groups=8
  export reward_num_layers=80
  export reward_hidden_size=8192
  export reward_num_attention_heads=64
  export reward_ffn_hidden_size=28672
  export reward_num_query_groups=8
  export group_query_attention=True
fi
