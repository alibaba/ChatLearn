set -x
nnodes=1
nproc_per_node=8
node_rank=0
master_addr="127.0.0.1"
master_port="9001"
world_size=8
script_and_args=$1
cmd_args="--nproc_per_node=$nproc_per_node \
    --nnodes=$nnodes \
    --node_rank=$node_rank \
    --master_addr=$master_addr \
    --master_port=$master_port  $script_and_args \
"

export PYTHONPATH=../

END=$nproc_per_node
world_size=$(( nproc_per_node * nnodes ))
for ((i=0;i<END;i++)); do
    echo $i
    rank=$(( node_rank * 8 + i ))
    LOCAL_RANK=$i GPU_NUM_DEVICES=$nproc_per_node RANK=$rank WORLD_SIZE=${world_size} MASTER_ADDR=${master_addr} MASTER_PORT=${master_port} python $script_and_args &
done

