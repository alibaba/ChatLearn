# Profile

ChatLearn 提供了两种 Profile 的方式：
1. torch profiler
2. nsys

注意：对于大模型，profile 的结果会非常大，建议在 profile 的时候减小模型尺寸。

## Torch Profiler

用户可以在执行脚本中配置 `runtime_args.profiler_dir=path_to_profile_dir` 来开启 Torch profiler。

```bash
runtime_args.profiler_dir=path_to_profile_dir
```

## nsys

用户可以在执行脚本中配置 `runtime_args.nsys=true` 来开启 nsys 的 profiler。

```bash
runtime_args.nsys=true
```

在启动程序的时候，需要在执行命令前加上 nsys 的启动参数，可以参考下述命令

```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace=true -x true --force-overwrite true -o my_profile \
python train_rlhf.py XXX
```

