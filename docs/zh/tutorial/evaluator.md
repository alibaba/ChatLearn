# Evaluator

本文档将介绍如何进行模型评估。用户可以使用 `EvalEngine` 单独对模型进行评估，也可以在训练 Engine 里配置 evaluator 在训练的过程中进行评估。

```python
def eval_flow(batch):
    p = policy.forward_step(batch)
    r = reward.eval_step(p)
    r1 = reward2.eval_step(p)
    return r, r1
evaluator = Evaluator(eval_flow)
evaluator.set_dataset(prompts)
results = evaluator.eval()
```
在上述例子中，我们构建了一个三个模型的评估flow，用户可以自定义 evaluation 的执行 flow。
evaluator.eval 返回的结果是一个 dict 类型，key 是 model_name，value 是一个 list，包含 batch 的计算结果。
在上述例子中，eval 返回的结果为 {"reward": [batch0, batch1, batch2], "reward2": [batch0, batch1, batch2]}
