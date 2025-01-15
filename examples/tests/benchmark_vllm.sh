#!/bin/bath
set -exo pipefail

[ -z "$ROOT" ] && export ROOT=path-to-root-dir
[ -z "$CHATLEARN" ] && export CHATLEARN=path-to-chatlearn
[ -z "$MODEL_NAME" ] && export MODEL_NAME="MoE"
[ -z "$HF_LOAD" ] && export HF_LOAD=path-to-hf-ckpt
[ -z "$DATASET_PATH" ] && export DATASET_PATH=path-to-dataset-json
[ -z "$LOG_ROOT" ] && LOG_ROOT=$ROOT/logs
[ -z "$TP_SIZE" ] && export TP_SIZE=1
[ -z "$PP_SIZE" ] && export PP_SIZE=1
[ -z "$NUM_PROMPTS" ] && export NUM_PROMPTS=16
[ -z "$NUM_SAMPLING" ] && export NUM_SAMPLING=8

# fix TypeError originated from get_device_name().replace() for vLLM v0.6.3
sed -i "291c \ \ \ \ device_name = str(current_platform.get_device_name()).replace(\" \", \"_\")" \
    /usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py


mkdir -p $LOG_ROOT/vllm_benchmark

benchmark_vllm(){
    use_async_llm_engine=${1:-"False"}
    tp_size=${2:-1}
    pp_size=${3:-1}
    sleep_time=${4:-"15m"}

    cd $CHATLEARN

    # start ray cluster
    ray stop
    export HEAD_NODE_ADDRESS=$MASTER_ADDR
    RAY_START_CMD="ray start "
    if [ ${RANK} == 0 ]; then
        RAY_START_CMD+=" --head --port=30010"
    else
        RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:30010"
    fi
    $RAY_START_CMD

    if [[ $use_async_llm_engine == "True" ]]; then
        LLM_ENGINE_ARGS="--use-async-llm-engine"
    else
        LLM_ENGINE_ARGS=""
    fi

    if  [ ${RANK} == 0 ]; then
        python examples/tests/benchmark_vllm.py \
            --model-name $MODEL_NAME \
            --model-path $HF_LOAD \
            --data $DATASET_PATH \
            --log-dir $LOG_ROOT/vllm_benchmark \
            --tensor-parallel-size $tp_size \
            --pipeline-parallel-size $pp_size \
            --num-prompts $NUM_PROMPTS \
            --num-sampling $NUM_SAMPLING \
            ${LLM_ENGINE_ARGS}
    else
        sleep ${sleep_time}
    fi
}

benchmark_vllm True $TP_SIZE $PP_SIZE
