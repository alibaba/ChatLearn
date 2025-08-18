# Resume Training and Fault Tolerance

Reinforcement learning tasks involve computations and interactions among multiple models. As model sizes grow and computational resources increase, occasional failures in the underlying software stack or hardware environment may cause tasks to stop unexpectedly. To ensure interrupted tasks can recover their state and automatically resume, ChatLearn provides a resume training capability.

## Configuring Resume Training in ChatLearn

You can enable resume training by setting `runtime_args.enable_resume_training=true`. ChatLearn will save relevant information and models to the directory specified by `runtime_args.output_dir`.

After a task interruption, you can simply re-run the same command to resume training from where it left off.

> Currently, this feature is enabled by default.