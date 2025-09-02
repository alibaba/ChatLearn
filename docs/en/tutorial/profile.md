# Profile
ChatLearn provides two ways to profile performance:
1. Torch profiler
2. nsys
Note: For large models, the profile result can be very large. It is recommended to reduce the model size when profiling.

## Torch Profiler
Users can enable the Torch profiler by configuring the rlhf setting `runtime_args.profiler_dir=path_to_profile_dir` in bash script.
```bash
runtime_args.profiler_dir=path_to_profile_dir
```

## nsys
Users can enable the nsys profiler by configuring the rlhf setting `runtime_args.nsys=true`  in bash script.
```yaml
runtime_args.nsys=true
```
When launching the program, nsys startup parameters need to be added before the execution command, as shown in the following example:
```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace=true -x true --force-overwrite true -o my_profile \
python train_rlhf.py XXX
```