# 数据集准备

本文以GRPO算法为示例，介绍如何准备ChatLearn所需的训练数据集。

可以参考[math_lighteval](https://github.com/alibaba/ChatLearn/blob/main/chatlearn/data/data_preprocess/math_lighteval.py)示例，按jsonl格式准备自定义数据集，每一行的json中需要包含如下必要元素：

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

- data_source

数据源信息，用于将数据路由到不同的reward函数。[link](https://github.com/alibaba/ChatLearn/blob/main/chatlearn/models/reward/rule_reward.py#L80)
> 自定义数据集需要实现相应的reward函数

- prompt

输入数据问题，按OpenAI范式的多轮对话格式组织。在数据处理过程中，会调用[transformers](https://github.com/huggingface/transformers)库的apply_chat_template接口将多轮对话拼接成LLM需要的prompt。

- reward_model

包含当前问题的正确答案，用于在reward函数判断答案的对错

数据集准备完成后，通过修改`runtime_args.data_path`、`runtime_args.eval_data_path`即可将自定义数据传入训练流程。

## 自定义奖励函数
你的奖励函数应放在 **ChatLearn/chatlearn/utils/rule_reward_score/** 目录下的一个 Python 文件 *customized_reward.py* 中，格式如下：

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

## 选择你的奖励函数
我们的 rule_reward 会根据 `data_source` 来选择对应的奖励函数（参见：[选择奖励函数](https://github.com/alibaba/ChatLearn/blob/5717ff4d15a249b79b570d4bbe4579b9d1af549e/chatlearn/models/reward/rule_reward.py#L80)）。
请确保在此处导入你自定义的奖励函数，例如：

```python
if data_source in [your_customized_dataset]:
    from chatlearn.utils.rule_reward_score import customized_reward
    return customized_reward.compute_score
```