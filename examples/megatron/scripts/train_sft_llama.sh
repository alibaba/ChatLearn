#!/bin/bash
set -x

[ -z "$MASTER_ADDR" ] && export MASTER_ADDR=localhost
[ -z "$WORLD_SIZE" ] && export WORLD_SIZE=1
[ -z "$GPUS_PER_NODE" ] && export GPUS_PER_NODE=8
[ -z "$RANK" ] && export RANK=0
[ -z "$MASTER_PORT" ] && export MASTER_PORT=12456

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

# check the path
[[ -z "${MEGATRON}" ]] && { echo "MEGATRON path is not set"; exit 1; }
[[ -z "${CHATLEARN}" ]] && { echo "CHATLEARN path is not set"; exit 1; }
[[ -z "${LOAD_PATH}" ]] && { echo "LOAD_PATH is not set"; exit 1; }
[[ -z "${TOKENIZER_MODEL}" ]] && { echo "TOKENIZER_MODEL is not set"; exit 1; }
[[ -z "${DATASET_PATH}" ]] && { echo "DATASET_PATH is not set"; exit 1; }


export PYTHONPATH=${PYTHONPATH}:${MEGATRON}:${CHATLEARN}/examples/megatron:${CHATLEARN}

[ -z "$model_size" ] && export model_size=llama2-7B
[ -z "$TOKENIZER_TYPE" ] && export TOKENIZER_TYPE=Llama2Tokenizer

if [ $model_size = llama2-7B ]; then
  NUM_LAYERS=32
  HIDDEN_SIZE=4096
  NUM_ATTN_HEADS=32
  INTERMEDIATE_SIZE=11008
  tp=4
  pp=1
elif [ $model_size = llama2-13B ]; then
  NUM_LAYERS=40
  HIDDEN_SIZE=5120
  NUM_ATTN_HEADS=40
  INTERMEDIATE_SIZE=13824
  tp=8
  pp=1
elif [ $model_size = llama2-70B ]; then
  NUM_LAYERS=80
  HIDDEN_SIZE=8192
  NUM_ATTN_HEADS=64
  INTERMEDIATE_SIZE=28672
  tp=8
  pp=4
  mb=2
  gbs=64
elif [ $model_size = llama3-8B ]; then
  NUM_LAYERS=32
  HIDDEN_SIZE=4096
  NUM_ATTN_HEADS=32
  INTERMEDIATE_SIZE=14336
  tp=4
  pp=1
  TOKENIZER_TYPE=Llama3Tokenizer
elif [ $model_size = llama3-70B ]; then
  NUM_LAYERS=80
  HIDDEN_SIZE=8192
  NUM_ATTN_HEADS=64
  INTERMEDIATE_SIZE=28672
  tp=8
  pp=4
  mb=2
  gbs=64
  TOKENIZER_TYPE=Llama3Tokenizer
fi

[ -z "$mb" ] && mb=8
[ -z "$gbs" ] && gbs=$((mb * 8))

seq_len=2048

DIR=$(pwd)
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
mkdir -p $DIR/logs

NODE_RANK=$RANK
NNODES=$WORLD_SIZE


dp=$(($WORLD_SIZE * $GPUS_PER_NODE / $tp / $pp))
gbs=$(($gbs * $dp))


[ -z "$CHECKPOINT_PATH" ] && CHECKPOINT_PATH=${CHATLEARN}/output/sft/hh_sft_$(date +%F)_gpt_${MODEL_SIZE}_${NNODES}w${GPUS_PER_NODE}g_tp${tp}_pp${pp}_mb${mb}_seqlen${seq_len}

mkdir -p $CHECKPOINT_PATH

MODEL_ARGS="
--max-position-embeddings 4096 \
--tokenizer-type ${TOKENIZER_TYPE} \
--tokenizer-model ${TOKENIZER_MODEL} \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--bf16 \
--untie-embeddings-and-output-weights \
--disable-bias-linear \
--use-rotary-position-embeddings \
--normalization RMSNorm \
--no-position-embedding \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32 "

use_legacy_models=${USE_LEGACY_MODELS:-"True"}

if [[ ${use_legacy_models} = "False" ]]; then
  MCORE_ARGS="--transformer-impl transformer_engine "
else
  if [[ ${MODEL_SIZE} = "llama3-8B" || ${MODEL_SIZE} = "llama3-70B" ]]; then
    echo "Llama3 models are not supported with USE_LEGACY_MODELS=True"
    exit 1
  fi
  MCORE_ARGS="
    --use-legacy-models \
    --transformer-impl local "
fi

log_file=$CHECKPOINT_PATH/stderr_$NODE_RANK.log

export CUDA_DEVICE_MAX_CONNECTIONS=1

cd ${CHATLEARN}/examples/megatron/sft

torchrun $DISTRIBUTED_ARGS \
  entry/train_sft.py \
  --tensor-model-parallel-size $tp \
  --pipeline-model-parallel-size $pp \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${INTERMEDIATE_SIZE} \
  --num-attention-heads ${NUM_ATTN_HEADS} \
  --seq-length $seq_len \
  --micro-batch-size $mb \
  --global-batch-size $gbs \
  --train-iters 1000 \
  --lr-decay-iters 1000 \
  --lr-warmup-iters 40 \
  --lr 2.0e-5 \
  --min-lr 6.0e-12 \
  --lr-decay-style cosine \
  --log-interval 1 \
  --eval-iters 10 \
  --eval-interval 1000 \
  --data-path $DATASET_PATH/train.jsonl $DATASET_PATH/train.jsonl $DATASET_PATH/train.jsonl \
  --save-interval 1000 \
  --save $CHECKPOINT_PATH \
  --load $LOAD_PATH \
  --tensorboard-log-interval 100 \
  --split 98,2,0 \
  --clip-grad 1.0 \
  --weight-decay 0. \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --init-method-std 0.006 \
  --tensorboard-dir $CHECKPOINT_PATH \
  --num-workers 8 \
  --no-load-rng \
  --no-load-optim \
  --swiglu \
  --log-timers-to-tensorboard \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --dataloader-type cyclic \
  --use-flash-attn \
  --use-distributed-optimizer \
  --sequence-parallel \
  --finetune \
  --distributed-timeout-minutes 60 \
  $MODEL_ARGS $MCORE_ARGS 2>&1 | tee -a ${log_file} ; exit ${PIPESTATUS[0]}
