import os
from safetensors import safe_open
import torch

def has_files_with_suffix(directory, extension):
    """
    检查指定目录及其子目录下是否有特定扩展名的文件。

    :param directory: 要检查的目录路径
    :param extension: 要检查的文件扩展名（例如 '.txt'）
    :return: 如果有特定扩展名的文件，返回 True；否则返回 False
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    if file_list != []:
        return file_list
    return False

if __name__ == "__main__":
    # 示例用法
    directory_to_check = '/mnt/nas/share2/home/qbc/transformers'
    file_extension = '.safetensors'

    # if has_files_with_suffix(directory_to_check, file_extension):
        # print(f"目录 {directory_to_check} 及其子目录下有 {file_extension} 文件。")
    # else:
        # print(f"目录 {directory_to_check} 及其子目录下没有 {file_extension} 文件。")
    
    file = has_files_with_suffix(directory_to_check, file_extension)
    if file != False:
        print(file)

def load_statedict(checkpoint):
    merged_state_dict = {}

    # 遍历每个文件并读取权重
    for file_path in checkpoint:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                merged_state_dict[key] = f.get_tensor(key)

    return merged_state_dict