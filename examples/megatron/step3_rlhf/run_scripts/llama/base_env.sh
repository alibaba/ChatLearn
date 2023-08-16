source run_scripts/base_env.sh
export PYTHONPATH=${PYTHONPATH}:${CHATLEARN}/examples/megatron/step2_reward

echo $PYTHONPATH
if ! python -c "from finetune_reward import model_provider"; then
   echo import failed
#    exit 1
fi


if [[ "$model_size" == "7B" ]]; then
  export policy_num_layers=32
  export policy_hidden_size=4096
  export policy_num_attention_heads=32
  export policy_ffn_hidden_size=11008
  export reward_num_layers=32
  export reward_hidden_size=4096
  export reward_num_attention_heads=32
  export reward_ffn_hidden_size=11008
elif [[ "$model_size" == "13B" ]]; then
  export policy_num_layers=40
  export policy_hidden_size=5120
  export policy_num_attention_heads=40
  export policy_ffn_hidden_size=13824
  export reward_num_layers=40
  export reward_hidden_size=5120
  export reward_num_attention_heads=40
  export reward_ffn_hidden_size=13824
fi