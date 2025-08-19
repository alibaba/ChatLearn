# 多节点分布式训练
## PAI DLC环境多节点分布式训练

ChatLearn已经适配了[PAI DLC](https://help.aliyun.com/zh/pai/user-guide/what-is-dlc?spm=a2c4g.11186623.help-menu-30347.d_3_8_0.348a6d12qLy7Nu&scm=20140722.H_2663875._.OR_help-T_cn~zh-V_1)分布式环境，您可以直接使用原来的单节点脚本进行多节点分布式强化学习训练。


> 您需要根据GPU总数来调整generation_batch_size、train_micro_batch_size等参数来达到最佳吞吐配置

## 自定义环境多节点分布式训练

如果在非DLC的环境中进行多节点训练，需要自行设置与分布式训练相关的环境。除了常见的分布式环境变量以为，ChatLearn中需要额外设置`LOCAL_MASTER_KEY=$MASTER_ADDR`。如下是一个2节点强化学习示例，在rank0和rank1节点中分别运行如下命令。

RANK0:

```bash
export MASTER_ADDR=你的主节点ip地址
export NNODES=2
export RANK=0
export LOCAL_MASTER_KEY=$MASTER_ADDR
# 执行强化学习训练
bash scripts/train_fsdp_vllm_qwen3_8b_grpo.sh
```

RANK1:

```bash
export MASTER_ADDR=你的主节点ip地址
export NNODES=2
export RANK=1
export LOCAL_MASTER_KEY=$MASTER_ADDR
# 执行强化学习训练
bash scripts/train_fsdp_vllm_qwen3_8b_grpo.sh
```