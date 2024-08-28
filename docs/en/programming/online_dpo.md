# Online DPO

This section will introduce the programming interface of ChatLearn. We will start with the main file and explain how to construct the `OnlineDPOEngine`. Then, we will discuss how to write models.


## Main Training File

The following is an example of a user's main training file:

1. Call `chatlearn.init()` to initialize the runtime environment for OnlineDPO.
2. Define the models required for training. Each model needs to have a unique `model_name`. Different models are distinguished by their `model_name` when configuring the model parameters. Please refer to the [training configuration](../config_yaml) for more details.
3. Define the engine [OnlineDPOEngine](../api/engine.rst).
4. Set up the training dataset.
5. Call `engine.learn` to start the OnlineDPO training.

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


## Model Definition

You can refer to [RLHF Programming Model definition](rlhf.md#model-definition) for model definition.


## Dataset

You can refer to [RLHF Programming Dataset](rlhf.md#dataset) to for dataset definition.
