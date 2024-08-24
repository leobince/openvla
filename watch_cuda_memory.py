import subprocess
import time
import re

def get_gpu_memory_usage(gpu_index=0):
    """使用nvidia-smi获取指定GPU的内存使用量"""
    command = f"nvidia-smi --query-gpu=memory.used --format=csv -i {gpu_index}"
    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    memory_usage = output.stdout
    try:
        # 解析内存使用量，通常形如 "memory.used [MiB]\n529 MiB\n"
        used_memory = int(re.findall(r"(\d+) MiB", memory_usage)[0])
    except IndexError:
        # 如果解析失败，返回0
        used_memory = 0
    return used_memory

def monitor_gpu_memory(duration=60, interval=1, gpu_index=0):
    """监控GPU内存使用情况，记录并返回最大使用量"""
    max_memory_used = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        current_memory_used = get_gpu_memory_usage(gpu_index)
        if current_memory_used > max_memory_used:
            max_memory_used = current_memory_used
        time.sleep(interval)  # 等待一段时间再次检测
    return max_memory_used

# 调用函数，监控1分钟内的最大内存使用
max_usage = monitor_gpu_memory(duration=60, interval=1, gpu_index=2)
print(f"Maximum memory used on GPU : {max_usage} MiB")