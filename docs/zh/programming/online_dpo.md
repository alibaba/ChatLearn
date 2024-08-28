# OnlineDPO

本章节将介绍 ChatLearn 的编程接口，我们会从主文件开始介绍如何构造 `OnlineDPOEngine`，然后再介绍如何编写模型。

## 训练主文件
以下为用户的训练主文件的范例。

1. 调用`chatlearn.init()`初始化 online_dpo 的运行环境。
2. 定义训练所需的模型。其中每个模型需要定义一个唯一的`model_name`。在配置模型参数的时候，不同模型的配置通过`model_name`来区分。详见[训练配置文件](../config_yaml)。
3. 定义 engine [OnlineDPOEngine](../api/engine.rst)。
4. 设置训练数据集。
5. 调用`engine.learn`开启 OnlineDPO 的训练。 


```python
from examples.megatron.models import PolicyInference
from examples.megatron.models import PolicyReference
from examples.megatron.models import PolicyTrainer
from examples.megatron.models import RewardInference

import chatlearn
from chatlearn import OnlineDPOEngine

# init
chatlearn.init()

# define models
policy_model = PolicyInference("policy")
reference_model = PolicyReference("reference")
reward_model = RewardInference("reward")
ppo_policy_model = PolicyTrainer("ppo_policy")

# define engine
engine = OnlineDPOEngine(policy_model,
                         reference_model,
                         reward_model,
                         ppo_policy_model)

# set dataset
train_prompts = ["test"] * 4096
engine.set_dataset(train_prompts)

# start online_dpo training
engine.learn()
```


## 模型定义

OnlineDPO训练模型定义和RLHF一致，可参考[RLHF Programming模型定义](rlhf.md#模型定义)章节。

## Dataset定义

OnlineDPO Dataset定义和RLHF一致，可参考[RLHF Programming Dataset](rlhf.md#dataset)章节。
