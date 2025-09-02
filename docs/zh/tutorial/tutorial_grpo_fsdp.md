# 基于 FSDP 的端到端GRPO训练流程

本文档提供使用 ChatLearn、PyTorch FSDP 和 vLLM 框架来对Qwen3模型进行GRPO训练的快速开始指南。

## 环境配置
1. Docker镜像准备
我们建议在PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC](https://help.aliyun.com/zh/pai/user-guide/create-a-training-task?spm=a2c4g.11186623.help-menu-30347.d_3_3_5_5.2dfb1925l3QjwG)中运行该示例，你需要填写如下镜像地址来启动实例：
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

可以使用vpc地址来加速镜像拉取速度，需要根据当前region信息来更改镜像地址。比如，启动在上海的DSW实例，可以使用如下镜像`dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312`。

2. 代码准备

```bash
git clone https://github.com/alibaba/ChatLearn.git && cd ChatLearn
```

## 数据准备
以[MATH-lighteval](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-lighteval)数据集作为示例.
```bash
# 下载数据集
mkdir -p dataset
modelscope download --dataset AI-ModelScope/MATH-lighteval --local_dir dataset/MATH-lighteval
# 数据集预处理
python chatlearn/data/data_preprocess/math_lighteval.py --input_dir dataset/MATH-lighteval --local_dir dataset/MATH-lighteval
```

## 训练
运行以下命令开始训练：

### Qwen3-8B
8卡机器运行如下命令
```bash
# 下载模型权重
modelscope download --model Qwen/Qwen3-8B --local_dir pretrained_models/Qwen3-8B
bash scripts/train_fsdp_vllm_qwen3_8b_grpo.sh
```

## 使用 Wandb 监控
如需使用 Wandb 记录训练过程，请修改对应脚本中的配置：

```bash
export WANDB_API_KEY="Your-Wandb-api-key"
```
将配置项改为：
```bash
runtime_args.log_args_dict.enable_wandb=True
runtime_args.log_args_dict.wandb_project="Your-Wandb-Project-Name"
```

## 模型转化
FSDP模型保存耗时较高，Chatlearn提供了离线模型转化功能，将FSDP保存的切片模型转化回huggingface模型。脚本如下：
```bash
export CHATLEARN=$(pwd)
python chatlearn/offline_ckpt_converter.py \
    --hf_dir ${CHATLEARN}/Qwen3-8B/ \
    --ckpt_dir ${CHATLEARN}/output/qwen3-grpo-8b/save_model/policy_trainer \
    --save_dir ${CHATLEARN}/output/qwen3-grpo-8b/save_model/huggingface/ \
    --iter 200 \
    --groupgemm 0
```
如果你使用groupgemm优化的moe模型训练，请确保设置：
```bash
   --groupgemm 1
```
这段脚本会将训练完成后的最后一个FSDP切片模型转化回HF模型，并保存在"${CHATLEARN}/output/qwen3-grpo-8b/save_model/huggingface/"路径下

## FAQ
### 如何可以加快PolicyTrainer的训练速度？
1. 设置models.policy_trainer.packing=True，并设置models.policy_trainer.max_token_in_packing=可以打满显存的总token数。
2. 如果是qwen3-moe模型，可以设置models.policy_trainer.groupgemm=True，打开groupgemm patch，提升moe层训练速度。
### 为什么FSDP初始化时，在transfomers读权重文件的时候ray有oom报错？
可以打开models.policy_trainer.meta_init=True。这可能会带来额外的初始化时间
### 为什么推理到一半会出现这个报错？
```bash
ray.exceptions.RayChannelTimeoutError: System error: If the execution is expected to take a long time, increase RAY_CGRAPH_get_timeout which is currently 10 seconds. Otherwise, this may indicate that the execution is hanging.
```
检查模型入参：models.policy.tensor_model_parallel_size是否不为1，若tensor_model_parallel_size不为1，建议设置models.policy.enforce_eager=True。
### 为什么在训练时会出现，torch.OutOfMemoryError: CUDA out of memory？
1. 如果models.policy_trainer.packing=True，尝试降低models.policy_trainer.max_token_in_packing。
2. 如果models.policy_trainer.packing=False，尝试降低runtime_args.train_micro_batch_size。
3. 如果runtime_args.train_micro_batch_size=1，或models.policy_trainer.max_token_in_packing小于生成长度依然oom，则建议提高models.policy_trainer.ulysses_sequence_parallel_size，推荐设置为2的倍数，且不超过单机卡数。
### 为什么做了如上调整，依然会torch.OutOfMemoryError: CUDA out of memory？
建议增加卡数，FSDP的显存消耗与总卡数近似是线性关系
### 为什么vllm初始化的时候会torch.OutOfMemoryError: CUDA out of memory？
适当提高models.policy.gpu_memory_utilization，最好不要超过0.95
