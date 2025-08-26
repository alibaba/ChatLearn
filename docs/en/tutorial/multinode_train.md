# Multi-Node Distributed Training

## Multi-Node Distributed Training in PAI DLC Environment

ChatLearn has been adapted to the [PAI DLC](https://help.aliyun.com/zh/pai/user-guide/what-is-dlc?spm=a2c4g.11186623.help-menu-30347.d_3_8_0.348a6d12qLy7Nu&scm=20140722.H_2663875._.OR_help-T_cn~zh-V_1) distributed environment, allowing you to directly use your original single-node scripts for multi-node distributed reinforcement learning training.

> You need to adjust parameters such as `generation_batch_size` and `train_micro_batch_size` based on the total number of GPUs to achieve optimal throughput configuration.

## Multi-Node Distributed Training in Custom Environments

If performing multi-node training in a non-DLC environment, you need to manually set up the environment variables related to distributed training. In addition to common distributed environment variables, ChatLearn requires the additional setting `LOCAL_MASTER_KEY=$MASTER_ADDR`. Below is a two-node reinforcement learning example. Run the following commands respectively on the rank0 and rank1 nodes.

RANK0:

```bash
export MASTER_ADDR=your_master_node_ip_address
export NNODES=2
export RANK=0
export LOCAL_MASTER_KEY=$MASTER_ADDR
# Execute reinforcement learning training
bash scripts/train_fsdp_vllm_qwen3_8b_grpo.sh
```

RANK1:

```bash
export MASTER_ADDR=your_master_node_ip_address
export NNODES=2
export RANK=1
export LOCAL_MASTER_KEY=$MASTER_ADDR
# Execute reinforcement learning training
bash scripts/train_fsdp_vllm_qwen3_8b_grpo.sh
```