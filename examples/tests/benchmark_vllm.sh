#!/bin/bash
set -exo pipefail

[ -z "$ROOT" ] && export ROOT=path-to-root-dir
[ -z "$CHATLEARN" ] && export CHATLEARN=path-to-chatlearn
[ -z "$MODEL_NAME" ] && export MODEL_NAME="MoE"
[ -z "$HF_LOAD" ] && export HF_LOAD=path-to-hf-ckpt
[ -z "$DATASET_PATH" ] && export DATASET_PATH=path-to-dataset-json
[ -z "$LOG_ROOT" ] && export LOG_ROOT=$ROOT/logs
[ -z "$WORLD_SIZE" ] && export WORLD_SIZE=1
[ -z "$GPUS_PER_NODE" ] && export GPUS_PER_NODE=8

[ -z "$USE_ASYNC_LLM_ENGINE" ] && export USE_ASYNC_LLM_ENGINE="False"
[ -z "$USE_CUDA_GRAPH" ] && export USE_CUDA_GRAPH="True"
[ -z "$USE_RAY_SPMD_WORKER" ] && export USE_RAY_SPMD_WORKER="False"
[ -z "$TP_SIZE" ] && export TP_SIZE=1
[ -z "$PP_SIZE" ] && export PP_SIZE=1
[ -z "$NUM_PROMPTS" ] && export NUM_PROMPTS=16
[ -z "$NUM_SAMPLING" ] && export NUM_SAMPLING=8
[ -z "$NUM_SCHEDULER_STEPS" ] && export NUM_SCHEDULER_STEPS=1
[ -z "$TIMEOUT" ] && export TIMEOUT="15"


# fix TypeError originated from get_device_name().replace() for vLLM v0.6.3
sed -i "291c \ \ \ \ device_name = str(current_platform.get_device_name()).replace(\" \", \"_\")" \
    /usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py

mkdir -p $LOG_ROOT/vllm_benchmark

benchmark_vllm(){
    use_async_llm_engine=${1:-"False"}
    use_cuda_graph=${2:-"True"}
    use_ray_spmd_worker=${3:-"False"}
    tp_size=${4:-1}
    pp_size=${5:-1}
    num_prompts=${6:-16}
    num_sampling=${7:-8}
    num_scheduler_steps=${8:-1}
    timeout=${9:-"15"}

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

    if [[ $use_cuda_graph == "True" ]]; then
        CUDA_GRAPH_ARGS=""
    else
        CUDA_GRAPH_ARGS="--enforce-eager"
    fi

    if [[ $use_ray_spmd_worker == "True" ]]; then
        SPMD_ARGS="--vllm-use-ray-spmd-worker"
    else
        SPMD_ARGS=""
    fi

    if  [[ ${RANK} == 0 ]]; then
        python examples/tests/benchmark_vllm.py \
            --model-name $MODEL_NAME \
            --model-path $HF_LOAD \
            --data $DATASET_PATH \
            --log-dir $LOG_ROOT/vllm_benchmark \
            --tensor-parallel-size $tp_size \
            --pipeline-parallel-size $pp_size \
            --num-prompts $num_prompts \
            --num-sampling $num_sampling \
            --num-scheduler-steps $num_scheduler_steps \
            ${LLM_ENGINE_ARGS} ${CUDA_GRAPH_ARGS} ${SPMD_ARGS}
    fi

    # Barrier all ranks on min(timeout, time(benchmark_vllm))
    torchrun --nnodes=$WORLD_SIZE --nproc_per_node=$GPUS_PER_NODE \
        examples/tests/barrier.py \
        --timeout $timeout
}

# Insert barrier here to ensure all workers have been succesfully started
torchrun --nnodes=$WORLD_SIZE --nproc_per_node=$GPUS_PER_NODE \
    examples/tests/barrier.py

benchmark_vllm \
    $USE_ASYNC_LLM_ENGINE \
    $USE_CUDA_GRAPH \
    $USE_RAY_SPMD_WORKER \
    $TP_SIZE \
    $PP_SIZE \
    $NUM_PROMPTS \
    $NUM_SAMPLING \
    $NUM_SCHEDULER_STEPS \
    $TIMEOUT
