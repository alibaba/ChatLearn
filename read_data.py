import os
import filecmp
import torch
from tqdm import tqdm

def list_files(directory):
    """列出目录中的所有文件"""
    return set(os.listdir(directory))

def compare_files(file1, file2):
    """比较两个文件是否相同"""
    return filecmp.cmp(file1, file2, shallow=False)

def compare_directories(dir1, dir2):
    """比较两个目录中的文件"""
    files1 = list_files(dir1)
    files2 = list_files(dir2)

    if files1 != files2:
        print("两个目录中的文件数量或名称不一致。")
        return

    print("文件数量一致，开始比较文件内容...")
    all_match = True
    for filename in files1:
        # filename = 'visual.blocks.26.attn.qkv.weight'
        file1 = os.path.join(dir1, filename)
        file2 = os.path.join(dir2, filename)

        # megatron_tensor = torch.load(file1, map_location="cpu")
        # fsdp_tensor = torch.load(file2, map_location="cpu")
        # shape = fsdp_tensor.shape
        # if not (megatron_tensor==fsdp_tensor).sum().item()==torch.prod(torch.tensor(shape)).item():
        #     print(f"文件 {filename} 内容不同。")
        #     all_match = False

        if not compare_files(file1, file2):
            print(f"文件 {filename} 内容不同。")
            all_match = False

        # import pdb
        # pdb.set_trace()
     
    if all_match:
        print("所有同名文件内容均相同。")

# 定义目录路径
megatron_params_dir = '/mnt/data/xinyi.zxy/chatlearn_dev/zxy_dev/ChatLearn/output/debug_sync_parameters-mcore-0922/after_sync_parameter/0'
fsdp_params_dir = '/mnt/data/xinyi.zxy/chatlearn_dev/zxy_dev/ChatLearn/output/debug_sync_parameters_fsdp/after_sync_parameter/0'

# 执行比较
compare_directories(megatron_params_dir, fsdp_params_dir)

