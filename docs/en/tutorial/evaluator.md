# Evaluator

This document will introduce how to perform model evaluation. Users can use `EvalEngine` to evaluate models independently or configure the evaluator within the training engine to perform evaluations during training.

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

In the above example, we constructed an evaluation flow for three models. Users can customize the evaluation execution flow through the `eval_flow`.

The result returned by `evaluator.eval` is of type `dict`, where the key is `model_name` and the value is a `list` containing the results of the computations for each batch.

In the above example, the result returned by `eval` will be `{"reward": [batch0, batch1, batch2], "reward2": [batch0, batch1, batch2]}`.
