# DPO

本章节将介绍 ChatLearn 的编程接口，我们会从主文件开始介绍如何构造 `DPOEngine`，然后再介绍如何编写模型。

## 训练主文件
以下为用户的训练主文件的范例。

1. 调用`chatlearn.init()`初始化 dpo 的运行环境。
2. 定义训练所需的模型。其中每个模型需要定义一个唯一的`model_name`。在配置模型参数的时候，不同模型的配置通过`model_name`来区分。详见[训练配置文件](../config_yaml)。
3. 定义 engine [DPOEngine](../api/engine.rst)。
4. 设置训练数据集。
5. 调用`engine.learn`开启 DPO 的训练。 


```python
from examples.megatron.models import PolicyReference
from examples.megatron.models import PolicyTrainer

import chatlearn
from chatlearn import DPOEngine

# init
chatlearn.init()

# define models
reference_model = PolicyReference("reference")
ppo_policy_model = PolicyTrainer("ppo_policy")

# define engine
engine = DPOEngine(reference_model,
                   ppo_policy_model)

# set dataset
train_prompts = ["test"] * 4096
engine.set_dataset(train_prompts)

# start dpo training
engine.learn()
```


## 模型定义

用户的模型需要继承`BaseModule`或其子类，`TorchModule`为通用的 Torch 模型的封装，`MegatronModule`为 Megatron 模型的封装。如果用户的 DPO 建模是基于 Megatron-LM，可以直接继承`MegatronModule`完成模型的建模。以继承`MegatronModule`为例，下述两段代码展现了 reference 模型的建模和 policy trainer 模型建模的例子：
1. 对于 reference 模型，用户需要实现`setup`和`forward_step`方法。在`setup`中，完成模型的定义，参数初始化，全局参数定义等工作。在`forward_step`中，实现模型一次前向所需的逻辑。
2. 对于 policy trainer 模型，用户需要实现`setup`和`train_step`方法。在`train_step`中，实现训练一个 step 所需的逻辑。
3. 除此之外，PolicyReference 模型需要实现`build_dataset`方法，完成 prompt 数据集的构建。

更多 API 信息参考[RLHF Module API](../api/module.rst).

```python
from chatlearn import MegatronModule


class PolicyReference(MegatronModule):

    def __init__(self, name):
        """
        Args:
            name: model name
        """

    def setup(self):
        """
        1. define model, self.model = xxx
        2. init global variables, etc.
        3. for training model, define optimizer, self.optimizer = xxx
        4. init model parameters
        """
        pass

    def forward_step(self, data, iteration=0):
        """
        Perform forward step for one batch
        Args:
            data: one batch for forward_step, type is dict
            iteration: iteration id for current step
        Returns:
            k/v dict
        """
        pass

    def build_dataset(self, train_prompts, is_eval=False):
        """
        Build prompt dataset. The implementation of build_dataset is exclusive to PolicyInference, whereas other models are not required to adopt it.

        Args:
            train_prompts: prompts provided by DPOEngine.set_dataset(train_prompts)
            is_eval: eval mode
        Returns:
            torch.utils.data.Dataset with user-defined collate_fn (see `Dataset`)
        """
        pass
```

```python
from chatlearn import MegatronModule


class PolicyTrainer(MegatronModule):

    def setup(self):
        """
        1. define model, self.model = xxx
        2. init global variables, etc.
        3. for training model, define optimizer, self.optimizer = xxx
        4. init model parameters
        """
        pass

    def train_step(self, data, iteration):
        """
        Perform train_step for one batch, including a list of micro-batches
        Args:
            data: one global batch for train_step, type is a list of dict, each dict is a micro-batch
            iteration: iteration id for current step
        """
        pass
```

## Dataset

DPO Dataset定义和RLHF一致，可参考[RLHF Programming Dataset](rlhf.md#dataset)章节。
