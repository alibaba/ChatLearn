# 环境和代码准备

1. 镜像准备

推荐参考 `https://github.com/alibaba/ChatLearn/tree/master/docker/ngc/Dockerfile.ngc22.10` 准备镜像。

2. 代码准备: 用户需要下载 `ChatLearn` 框架代码。

```
# 下载 ChatLearn 代码
git clone https://github.com/alibaba/ChatLearn.git
```

3. 如果您需要运行基于 Megatron-LM 框架的 RLHF 训练程序，您也需要下载为支持ChatLearn训练修改后的 `Megatron-LM-ChatLearn` 代码。

```
# 下载 Megatron-LM-ChatLearn
git clone -b chatlearn-2308 https://github.com/alibaba/Megatron-LM-ChatLearn.git
```