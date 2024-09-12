# 配置文件
## 训练配置文件

用户需要一个程序主 yaml 配置来设置运行环境、模型配置和 RLHF 训练流程相关的配置。同时，用户也可能需要为每个模型配置单独的模型配置。

RLHF 的训练配置包括三部分

1. runtime_env: 运行环境配置
2. models: 模型配置。每一个模型都可以单独配置模型参数。通过`model_name`来区分不同的模型。这里`model_name`对应主文件中定义模型时传入的`model_name`。
3. runtime: 训练配置

以下为一个训练配置的示例。具体的配置项含义可以参考 [Config API 文档](api/config.rst).

为了方便配置不同的超参数，我们也支持从环境变量读取参数。格式如下

```
param: ${env_name:default_value}
```
`param`为参数名，`env_name`为环境变量名，`default_value`为默认值 (可选)。在以下例子中，如果设置了环境变量`ref_generation_batch_size`, 则会从环境变量中读取赋值给`reference`的`generation_batch_size`，如果没有设置环境变量`ref_generation_batch_size`，则使用默认值 4。

```yaml
runtime_env:
  platform: DLC
  excludes:
    - "*pt"
    - "logs"
    - "tensorboards"
    - ".nfs*"


models:
  policy:
    model_config_file: policy_inference.yaml
    num_gpu: 8
    trainable: False

  reference:
    model_config_file: reference.yaml
    num_gpu: 8
    trainable: False
    generation_batch_size: ${ref_generation_batch_size:4}

  reward:
    model_config_file: reward_inference.yaml
    num_gpu: 8
    trainable: False

  value:
    model_config_file: old_value_inference.yaml
    num_gpu: 8
    trainable: False

  ppo_policy:
    model_config_file: ppo_policy.yaml
    num_gpu: 8
    trainable: True

  ppo_value:
    model_config_file: ppo_value.yaml
    num_gpu: ${num_gpu}
    trainable: True

runtime:
  colocation:
    - policy,ppo_policy,reward,reference,value,ppo_value
  generation_batch_size: ${generation_batch_size:4}
  train_micro_batch_size: 2
  train_global_batch_size: ${train_global_batch_size:512}
  num_episode: 200
  sample_per_episode: ${sample_per_episode:1024}
  num_training_epoch: 1
  save_episode_interval: ${save_episode_interval:50}
  data_path: ${data_path}
  eval_episode_interval: ${eval_episode_interval:100}
```


## 模型配置 YAML

本框架支持对每个模型配置单独的配置文件，用于配置不同模型的超参数，并行化策略，checkpoint 初始化等。模型配置文件格式为 yaml 文件。下面是一个简单的模型配置例子。

```yaml
num_layers: 6
hidden_size: 768
num_attention_heads: 12
bf16: True
seq_length: 2048
tensor_model_parallel_size: 8
pipeline_model_parallel_size: 2
load: path-to-ckpt
```

为了简化不同模型的共享配置，我们拓展了 yaml 的语法，通过 include 的字段来集成 base 配置文件的配置。在下面这个例子中，`policy_inference.yaml`和`ppo_policy.yaml`共享`num_layers`/`hidden_size`等参数，同时，两个模型的配置了各自不同的`pipeline_model_parallel_size`。

![yaml](../images/yaml.jpg)
