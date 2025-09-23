# Building an Agent Training Pipeline with ChatLearn

This document provides an end-to-end example of building an Agent reinforcement learning pipeline using the ChatLearn, PyTorch FSDP, SGLang, and LangGraph frameworks.

## Environment Setup

ChatLearn currently supports building custom Agent workflows using SGLang + LangGraph for end-to-end agentic reinforcement learning training.

1. Docker Image Preparation
We recommend running this example in PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/) or [DLC]( https://help.aliyun.com/zh/pai/user-guide/create-a-training-task). You need to specify the following image address to launch the instance:

```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.8.0-sglang0.5.2-ubuntu24.04-cuda12.6-py312
```

You can use the VPC address to accelerate image pulling speed, which requires modifying the image address according to your current region. For example, to launch a DSW instance in Shanghai, you can use the following image:
`dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.8.0-sglang0.5.2-ubuntu24.04-cuda12.6-py312`.

2. Code Preparation

```bash
git clone https://github.com/alibaba/ChatLearn.git && cd ChatLearn
```

## Data Preparation
Using the [MATH-lighteval](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-lighteval) dataset as an example.
```bash
# Download the dataset
mkdir -p dataset
modelscope download --dataset AI-ModelScope/MATH-lighteval --local_dir dataset/MATH-lighteval
# Preprocess the dataset
python chatlearn/data/data_preprocess/math_lighteval_agent.py --input_dir dataset/MATH-lighteval --local_dir dataset/MATH-lighteval
```

## Training

When performing agentic training, you need to configure the following parameters:

```bash
runtime_args.task_type=agent # Task type
runtime_args.rollout_backend=sglang # Rollout backend engine; agent tasks only support sglang
runtime_args.use_rollout_manager=True # Use rollout manager to manage async requests
models.policy.is_sync_mode=False # Rollout backend engine uses async mode (only supported by sglang)
```

Run the following command on an 8-GPU machine:
```bash
# Download model weights
modelscope download --model Qwen/Qwen3-8B --local_dir pretrained_models/Qwen3-8B
bash scripts/fsdp_sglang/train_fsdp_sglang_qwen3_8b_grpo_agentic_math_task.sh
```

## Customizing the Learning Workflow

ChatLearn uses LangGraph to build Agent workflows. To construct a custom Agent training process, you need to implement the following three components:

1. Custom Dataset

Refer to the example file [math_lighteval_agent](../../../chatlearn/data/data_preprocess/math_lighteval_agent.py) to prepare your dataset. If your task dataset depends on additional information, you can add it in [prompt_dataset](../../../chatlearn/data/prompt_dataset.py), which will be passed through ChatLearn's execution graph.

Compared to Chat task datasets, you need to include two additional fields in the dictionary: `agent_name` and `agent_cfg_path`.

- `agent_name`: Used to construct a custom AgentGraph and route corresponding data to the appropriate graph during Rollout sampling.
- `agent_cfg_path`: Path to the AgentGraph configuration file, used to configure any custom parameters for the AgentGraph. During training, you can access any parameter in this YAML file via `cfg.xxx`.

2. Custom AgentGraph

Refer to the example file [math_eval_agent_graph](../../../chatlearn/models/agent/examples/math_eval_agent_graph.py) to build a custom AgentGraph. You need to implement the following core components:

- **Registration**

Register the `agent_name` in `_graph_registry`. The agent graph will later be constructed and routed using this `agent_name`.
```
@register("agent_name")
```

- **Implementation of the `build_graph` function**

Implement a custom Agent workflow based on LangGraph. Refer to [this link](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) to understand LangGraph concepts.

- **Implementation of the `run` execution function**

This is an async function and serves as the specific entry point for executing the AgentGraph. Information passed through `prompt_dataset` for each data sample will be retained in `kwargs` and can be accessed as needed.

3. Agent Configuration YAML File (Optional)

Refer to [math_eval.yaml](../../../template/agent/math_eval.yaml) to configure any custom attributes for the AgentGraph in your experiment. The content of the YAML file is stored in the `AgentGraph.cfg` attribute.