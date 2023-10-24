source run_scripts/base_env.sh
export PYTHONPATH=${PYTHONPATH}:${CHATLEARN}/examples/megatron/step2_reward

echo $PYTHONPATH


if [[ "$model_size" == "1B1" ]]; then
  export policy_num_layers=24
  export policy_hidden_size=1536
  export policy_num_attention_heads=16
  export policy_ffn_hidden_size=6144
  export reward_num_layers=24
  export reward_hidden_size=1536
  export reward_num_attention_heads=16
  export reward_ffn_hidden_size=6144
elif [[ "$model_size" == "1B7" ]]; then
  export policy_num_layers=24
  export policy_hidden_size=2048
  export policy_num_attention_heads=16
  export policy_ffn_hidden_size=8192
  export reward_num_layers=24
  export reward_hidden_size=2048
  export reward_num_attention_heads=16
  export reward_ffn_hidden_size=8192
elif [[ "$model_size" == "7B1" ]]; then
  export policy_num_layers=30
  export policy_hidden_size=4096
  export policy_num_attention_heads=32
  export policy_ffn_hidden_size=16384
  export reward_num_layers=30
  export reward_hidden_size=4096
  export reward_num_attention_heads=32
  export reward_ffn_hidden_size=16384
fi
