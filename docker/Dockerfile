FROM registry.cn-hangzhou.aliyuncs.com/pai-dlc/pytorch-training:1.12-ubuntu20.04-py3.10-cuda11.3-llm


RUN pip install --no-cache-dir -U \
    ray[default]==2.3.0 \
    gpustat==1.0.0 \
    regex \
    flash-attn==1.0.4 \
    tokenizers \
    h5py \
    rjieba \
    wandb \
    transformers \
    datasets \
    cupy \
    torchtyping \
    protobuf==3.19.6 \
    http://odps-release.cn-hangzhou.oss.aliyun-inc.com/torchacc/torch_demo/datasets/gpt/bladnn-0.0.1-py3-none-any.whl


# prepare for dlc
RUN touch /usr/bin/prepare_dlc_environment
RUN chmod +x /usr/bin/prepare_dlc_environment
