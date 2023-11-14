[ -z "$MEGATRON" ] && export MEGATRON=path-to-megatron
[ -z "$CHATLEARN" ] && export CHATLEARN=path-to-chatlearn
[ -z "$TP" ] && export TP=4
[ -z "$model_type" ] && export model_type=llama
[ -z "$VOCAB_FILE" ] && export VOCAB_FILE=path-to-tokenizer
[ -z "$LOAD" ] && export LOAD=path-to-ckpt
[ -z "$DATASET_PATH" ] && export DATASET_PATH=path-to-dataset-json
[ -z "$model_size" ] && export model_size=13B
OUTPUT=$CHATLEARN/output/tests/
mkdir -p $OUTPUT

if [[ "$model_type" == "gpt" ]]; then
    configs=configs/gpt/test_policy.yaml
    export vocab_file=$VOCAB_FILE
    export merge_file=$MERGE_FILE
    export max_new_tokens=512
    export max_seq_len=1024
elif [[ "$model_type" == "llama2" ]]; then
    configs=configs/llama2/test_policy.yaml
    export tokenizer_model=$VOCAB_FILE
else
    echo "unexpected model_type $model_type."
    exit 1
fi

source run_scripts/$model_type/base_env.sh

export exp_name=run_test_${model_size}_tp${TP}_meg_$model_type
export batch_generation_min_prompt_length=32

generation_batch_size=64 \
num_device=$TP \
policy_tp=$TP \
eval_data_path=$DATASET_PATH \
policy_inference_load=$LOAD \
eval_output_dir=$OUTPUT \
python tests/test_policy_generation.py -c $configs 2>&1 | tee ${OUTPUT}/${exp_name}.log ; exit ${PIPESTATUS[0]}
