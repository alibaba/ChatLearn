# 使用ChatLearn构建Agent训练流程

本文档提供使用 ChatLearn、PyTorch FSDP 、SGLang、LangGraph 框架来构建端到端Agent强化学习示例。

## 环境配置

ChatLearn目前支持使用SGLang + LangGraph 搭建自定义Agent流程来进行端到端agentic强化学习训练。

1. Docker镜像准备
我们建议在PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC]( https://help.aliyun.com/zh/pai/user-guide/create-a-training-task)中运行该示例，你需要填写如下镜像地址来启动实例：
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.8.0-sglang0.5.2-ubuntu24.04-cuda12.6-py312
```

可以使用vpc地址来加速镜像拉取速度，需要根据当前region信息来更改镜像地址。比如，启动在上海的DSW实例，可以使用如下镜像`dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.8.0-sglang0.5.2-ubuntu24.04-cuda12.6-py312`。

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
python chatlearn/data/data_preprocess/math_lighteval_agent.py --input_dir dataset/MATH-lighteval --local_dir dataset/MATH-lighteval
```

## 训练

在进行agentic训练时，需要配置如下参数

```bash
runtime_args.task_type=agent # 任务类型
runtime_args.rollout_backend=sglang # rollout后端引擎，agent任务仅支持sglang
runtime_args.use_rollout_manager=True # 使用rollout manager管理async request
models.policy.is_sync_mode=False # rollout后端引擎使用async模式，仅sglang支持
```

8卡机器运行如下命令
```bash
# 下载模型权重
modelscope download --model Qwen/Qwen3-8B --local_dir pretrained_models/Qwen3-8B
bash scripts/fsdp_sglang/train_fsdp_sglang_qwen3_8b_grpo_agentic_math_task.sh
```

## 自定义学习流程

ChatLearn基于LangGraph来搭建Agent流程，需要实现以下三部分来构建自定义Agent训练。

1. 自定义数据集

可参考示例文件[math_lighteval_agent](../../../chatlearn/data/data_preprocess/math_lighteval_agent.py)准备数据集。如果任务数据集依赖其他额外信息，可在[prompt_dataset](../../../chatlearn/data/prompt_dataset.py)中进行添加，该信息会在ChatLearn的执行图中进行透传。

相比于Chat任务数据集，需要在字典中存放`agent_name`和`agent_cfg_path`两个信息。

- agent_name：用于构建自定义AgentGraph以及在Rollout采样过程中将对应的数据路由到对应的图。
- agent_cfg_path：AgentGraph参数文件路径，用于配置AgentGraph的任何自定义参数，在训练过程中可通过 cfg.xxx来获取该yaml文件中任意参数

2. 自定义AgentGraph

可参考示例文件[math_eval_agent_graph](../../../chatlearn/models/agent/examples/math_eval_agent_graph.py)构建自定义AgentGraph，需要实现如下几个核心部分。

- register注册

将agent_name注册到_graph_registry中，后续会通过agent_name进行agent graph的构建和路由
```
@register("agent_name")
```

- build_graph函数实现

基于LangGraph实现自定义Agent流程，可参考[link](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)了解LangGraph相关概念。

- run 执行函数实现

该函数为async函数，是AgentGraph的具体执行入口函数，其中每条数据集在prompt_dataset中传递的信息会在kwargs保留，可以按需获取。


3. agent_cfg yaml文件(可选)

参考[math_eval.yaml](../../../template/agent/math_eval.yaml)用于在实验中配置AgentGraph的任何自定义属性，yaml中的内容保存在AgentGraph.cfg属性中。
