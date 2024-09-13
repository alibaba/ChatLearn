# Efficient Memory Sharing (EMS)

ChatLearn provides EMS feature to significantly reduce the GPU memory usage during the alignment training. 
It maximizes the use of limited resources to train models with larger-scale or to improve overall training efficiency by improving the model's parallel strategy and increasing the batch size after GPU memory saved.

When multiple models in ChatLearn share the same resources for training or inference, enabling the EMS feature allows these models to sequentially share GPU memory:
- After each model is initialized, tensors/buffers that constantly reside in GPU memory (such as weights, gradient buffers, and optimization states) are unloaded to the RAM or freed to release their occupied GPU memory.
- Before training or inference for a specific model, the tensors/buffers are loaded from the RAM or reconstructed, and then training or inference takes place.
- Once the training or inference is complete, the tensors/buffers are again unloaded to the RAM or freed to release their occupied GPU memory.

By repeating the above process, multiple models sequentially share GPU memory, maximizing the efficiency of GPU memory usage.

## Usage
Users can specify whether to enable the EMS feature by configuring the `free_memory` (bool type, default is False) parameter for each model. This can be directly modified in the `rlhf.yaml` for each model. For example, to enable the EMS feature for the policy model:
```yaml
policy:
    model_config_file: old_policy_inference.yaml
    ...
    free_memory: ${free_memory_policy:True}
```
Alternatively, it can also be configured in the training script using environment variables:
- Policy model: `export free_memory_policy=True`
- Reference model: `export free_memory_reference=True`
- Reward model: `export free_memory_reward=True`
- Value model: `export free_memory_value=True`
- PPO policy model: `export free_memory_ppo_policy=True`
- PPO value model: `export free_memory_ppo_value=True`

A complete example can be found in the [llama2 configuration](https://github.com/alibaba/ChatLearn/blob/main/examples/megatron/configs/llama2/rlhf.yaml).