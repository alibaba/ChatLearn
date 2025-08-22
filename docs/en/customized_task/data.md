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