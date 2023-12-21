# Offload

随着模型规模变大，为了充分利用有限资源达到最佳训练性能，我们可以借助 Offload 的技术来减少训练过程中的显存占用，来增大 batch size 以提升整体的训练效率。
目前 ChatLearn 中支持了 Optimizer State Offload，未来我们会支持更多参数的 Offload。

## Optimizer State Offload
用户可以配置模型的 `offload_optimizer_states` (bool, 默认为 False) 参数来指定是否开启 Optimizer State Offload 。
如果 `offload_optimizer_states == True`, 将在模型执行前将 Optimizer State onload 到 GPU，并在模型执行完成 将 Optimizer State offload 
 到 CPU。

以下这个例子中，我们将对 `ppo_policy` 这个模型开启 Optimizer State Offload 。

```yaml
  ppo_policy:
    model_config_file: ppo_policy.yaml
    num_device: 8
    trainable: True
    offload_optimizer_states: True
```

完整示例可以参考 [llama2 配置](../../../examples/megatron/step3_rlhf/configs/llama2/rlhf.yaml)。
