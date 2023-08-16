# Modeling

```
from chatlearn import RLHFMegatronModule

class Policy(RLHFMegatronModule):


    def setup(self):
        """
        init model env / create model / optimizer / etc
        """
        pass
    

    def forward_step(self, data):
        """
        perform forward step for one batch
        Args:
            data: one batch for forward_step, type is dict
        Returns:
            k/v dict
        """
        pass


    def train_step(self, data, train_info):
        """
        Perform train_step for one batch, including a list of micro-batches
        Args:
            data: one global batch for train_step, type is a list of dict, each dict is a micro-batch
            train_info: includes training information, e.g., "iteration"
        """
        pass


    def save_checkpoint(self, iteration):
        """
        Save checkpoint given iteration.
        Args:
            iteration: current training iteration
        """
        pass

```

# Config

```
# base.yaml
num_layers: 6
hidden_size: 768
num_attention_heads: 12
max_position_embeddings: 2048
bf16: True
seq_length: 2048
```

```
includes:
        - base.yaml

tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1

load: xxx.pt
```

```
includes:
        - base.yaml

tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1

load: xxx.pt
lr: 0.00001
```
