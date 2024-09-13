# 高效显存复用（EMS）

ChatLearn 中提供高效显存复用 (Efficient Memory Sharing, EMS) 功能来大幅减少训练过程中的显存占用。 
EMS 功能可以充分利用有限资源来训练更大规模的模型，也可以利用节约的显存来调整模型的并行策略或者增大 batch size，从而提升整体的训练效率。

ChatLearn 中多个模型共享相同的资源进行训练或推理时，打开 EMS 功能，可以让这些模型按序共享使用显存：
- 每个模型初始化完成后，将常驻显存的各类 tensor/buffer（包括 weight, grad buffer, optim states 等）卸载到内存或者直接释放，清空该模型占用的显存；
- 某个模型训练或推理前，先从内存中加载或者重建 tensor/buffer，然后进行训练或推理；
- 训练或推理完成后，将常驻显存的 tensor/buffer 卸载到内存或者直接释放，再次清空该模型占用的显存。

重复如上流程，多个模型间按序共享使用显存，最大化显存利用效率。

## 功能用法
用户通过配置每个模型的 `free_memory` (bool 类型, 默认为 False) 参数来指定是否开启 EMS 功能。
可以直接修改 `rlhf.yaml` 中每个模型的 `free_memory` 配置，例如打开 policy 模型的 EMS 功能:

```yaml
policy:
    model_config_file: old_policy_inference.yaml
    ...
    free_memory: ${free_memory_policy:True}
```

用户也可以在训练脚本中通过配置环境变量来启动 EMS 功能：
- policy 模型：`export free_memory_policy=True`
- reference 模型：`export free_memory_reference=True`
- reward 模型：`export free_memory_reward=True`
- value 模型：`export free_memory_value=True`
- ppo_policy 模型：`export free_memory_ppo_policy=True`
- ppo_value 模型：`export free_memory_ppo_value=True`

完整示例可以参考 [llama2 配置](https://github.com/alibaba/ChatLearn/blob/main/examples/megatron/configs/llama2/rlhf.yaml)。
