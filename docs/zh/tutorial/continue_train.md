# 续跑和容错

强化学习任务涉及到多模型的计算和交互，随着模型规模的增大和计算资源的增加，由于依赖的软件栈和硬件环境都有可能出现偶发异常，会导致任务停止运行。
为了保障被中断的任务可以恢复状态进行自动续跑，ChatLearn提供了续跑的功能。

## 配置 ChatLearn 续跑

您可以通过配置`runtime_args.enable_resume_training=true`来打开续跑功能。ChatLearn会将相关信息及模型保存到`runtime_args.output_dir`中。

当任务中断后，可以直接运行相同命令续跑。

> 当前该开关已经默认开启
