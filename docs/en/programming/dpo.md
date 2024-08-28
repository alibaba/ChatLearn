# DPO

This section will introduce the programming interface of ChatLearn. We will start with the main file and explain how to construct the `DPOEngine`. Then, we will discuss how to write models.


## Main Training File

The following is an example of a user's main training file:

1. Call `chatlearn.init()` to initialize the runtime environment for DPO.
2. Define the models required for training. Each model needs to have a unique `model_name`. Different models are distinguished by their `model_name` when configuring the model parameters. Please refer to the [training configuration](../config_yaml) for more details.
3. Define the engine [DPOEngine](../api/engine.rst).
4. Set up the training dataset.
5. Call `engine.learn` to start the DPO training.

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


## Model Definition

User-defined models need to inherit from `BaseModule` or its subclasses. `TorchModule` is the wrapper for general Torch models, and `MegatronModule` is the wrapper for Megatron models. If the user's DPO modeling is based on Megatron-LM, they can directly inherit from `MegatronModule` to complete the model construction.

Here are examples of model construction for both the inference and training models, using inheritance from `MegatronModule`:
1. For the reference model, users need to implement the `setup` and `forward_step` methods. In `setup`, define the model, initialize parameters, and define global parameters. In `forward_step`, implement the logic required for one forward pass of the model.
2. For the training model, users need to implement the `setup` and `train_step` methods. In `train_step`, implement the logic required for one training step.
3. In addition, the `PolicyReference` model needs to implement the `build_dataset` method to construct the prompt dataset.

For more API information, refer to [RLHF Module API](../api/module.rst).

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

You can refer to [RLHF Programming Dataset](rlhf.md#dataset) to for dataset definition.
