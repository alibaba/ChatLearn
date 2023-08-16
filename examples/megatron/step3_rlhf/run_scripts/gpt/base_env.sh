source run_scripts/base_env.sh

[ -z "$num_device_policy" ] && export num_device_policy=$num_device
[ -z "$num_device_ref" ] && export num_device_ref=$num_device
[ -z "$num_device_reward" ] && export num_device_reward=$num_device
[ -z "$num_device_value" ] && export num_device_value=$num_device

[ -z "$sample_per_episode" ] && export sample_per_episode=2048
[ -z "$use_bladnn" ] && export use_bladnn=True
[ -z "$batch_generation_ranking" ] && export batch_generation_ranking=True

if [[ "$model_size" == "small" ]]; then
  export policy_num_layers=8
  export policy_hidden_size=1024
  export policy_num_attention_heads=16
  export reward_num_layers=8
  export reward_hidden_size=1024
  export reward_num_attention_heads=16
elif [[ "$model_size" == "30b" ]]; then
  export policy_num_layers=48
  export policy_hidden_size=7168
  export policy_num_attention_heads=56
  export reward_num_layers=48
  export reward_hidden_size=7168
  export reward_num_attention_heads=56
elif [[ "$model_size" == "13b" ]]; then
  export policy_num_layers=40
  export policy_hidden_size=5120
  export policy_num_attention_heads=40
  export reward_num_layers=40
  export reward_hidden_size=5120
  export reward_num_attention_heads=40
elif [[ "$model_size" == "7b" ]]; then
  export policy_num_layers=32
  export policy_hidden_size=4096
  export policy_num_attention_heads=32
  export reward_num_layers=32
  export reward_hidden_size=4096
  export reward_num_attention_heads=32
elif [[ "$model_size" == "66b" ]]; then
  export policy_num_layers=64
  export policy_hidden_size=9216
  export policy_num_attention_heads=72
  export reward_num_layers=64
  export reward_hidden_size=9216
  export reward_num_attention_heads=72
elif [[ "$model_size" == "175b" ]]; then
  export policy_num_layers=96
  export policy_hidden_size=12288
  export policy_num_attention_heads=96
  export reward_num_layers=96
  export reward_hidden_size=12288
  export reward_num_attention_heads=96
fi
