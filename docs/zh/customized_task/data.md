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




