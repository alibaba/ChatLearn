# Dataset Preparation

This document uses the GRPO algorithm as an example to illustrate how to prepare a training dataset for ChatLearn.

You can refer to the [math_lighteval](https://github.com/alibaba/ChatLearn/blob/main/chatlearn/data/data_preprocess/math_lighteval.py) example to prepare a custom dataset in JSONL format. Each line of the JSON file must contain the following required fields:

```json
{
  "data_source": "DigitalLearningGmbH/MATH-lighteval",
  "prompt": [
    {
      "content": "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have? Let's think step by step and output the final answer within \\boxed{}.",
      "role": "user"
    }
  ],
  "reward_model": {
    "ground_truth": "2",
    "style": "rule"
  }
}
```

- **data_source**

  Information about the data source, used to route data to different reward functions. [link](https://github.com/alibaba/ChatLearn/blob/main/chatlearn/models/reward/rule_reward.py#L80)  
  > For custom datasets, you need to implement the corresponding reward function.

- **prompt**

  The input question, formatted as a multi-turn conversation following the OpenAI paradigm. During data processing, the `apply_chat_template` interface from the [transformers](https://github.com/huggingface/transformers) library will be called to concatenate the conversation turns into a prompt suitable for the LLM.

- **reward_model**

  Contains the correct answer for the current question, which is used by the reward function to evaluate whether the model's response is correct.

After preparing the dataset, you can pass your custom data into the training pipeline by modifying `runtime_args.data_path` and `runtime_args.eval_data_path`.

# Customize Reward Function
For each dataset, you should implement a reward function to evaluate the generated responses. Several reward functions are already provided in the [reward_score directory]https://github.com/alibaba/ChatLearn/tree/update_doc/chatlearn/utils/rule_reward_score. You may also define custom reward functions as needed.

## Customized
Your reward function should be like this in a python file *customized_reward.py* under **ChatLearn/chatlearn/utils/rule_reward_score/**.:
```python
def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    This reward function evaluates solution_str against the provided ground_truth.

    Args:
        solution_str (str): Whole response generate by rollout engine.
        ground_truth (str): Pre-defined ground truth to evaluate solution_str.
    """
    ...
    return retval
```

## Select Your reward function
Our rule_reward actor choose reward funciton by data_source in here ([Choose reward function](https://github.com/alibaba/ChatLearn/blob/5717ff4d15a249b79b570d4bbe4579b9d1af549e/chatlearn/models/reward/rule_reward.py#L80)).
Make sure to import your customized reward function here like this:
```python
if data_source in [your_customized_dataset]:
    from chatlearn.utils.rule_reward_score import customized_reward
        return customized_reward.compute_score