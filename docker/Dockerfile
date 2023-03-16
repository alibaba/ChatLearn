FROM registry.cn-hangzhou.aliyuncs.com/pai-dlc/pai-pytorch-training:torch-1.12-cuda11.3-py38-acc-230219


RUN pip install --no-cache-dir -U \
    ray==2.3.0 \
    regex \
    flash-attn \
    tokenizers \
    h5py \
    rjieba \
    wandb \
    transformers \
    datasets \
    cupy \
    torchtyping

RUN echo NCCL_SOCKET_IFNAME=eth0 > /etc/nccl.conf

# prepare for dlc
RUN touch /usr/bin/prepare_dlc_environment
RUN chmod +x /usr/bin/prepare_dlc_environment