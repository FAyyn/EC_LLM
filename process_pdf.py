import os
import subprocess
import sys
import json

# 检查是否提供了目标目录和输出目录
if len(sys.argv) < 3:
    print("Usage: python script.py <target_directory> <output_directory> [<GPU_ID>]")
    sys.exit(1)

# 目标目录和输出目录
TARGET_DIR = os.path.abspath(sys.argv[1])
OUTPUT_DIR = os.path.abspath(sys.argv[2])

# 检查是否提供了GPU ID，默认使用 GPU 0
GPU_ID = sys.argv[3] if len(sys.argv) > 3 else 0

# 打印选择的GPU ID
print(f"Using GPU ID: {GPU_ID}")

# 设置CUDA_VISIBLE_DEVICES环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# 检查输出目录是否存在，如果不存在则创建
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# 检查输出目录是否包含完整的提取结果
def check_success(directory):
    # 确保路径中的空格和特殊字符不会干扰命令
    directory = os.path.abspath(directory)

    # 检查该目录下是否存在至少三个 .pdf 文件
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    if len(pdf_files) < 3:
        print(f"Missing PDF files in {directory}")
        return False

    # 检查该目录下是否有至少三个 .json 文件
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if len(json_files) < 3:
        print(f"Missing JSON files in {directory}")
        return False

    return True


# 遍历目标目录下的所有文件
# 使用 os.walk 和 glob 库，并确保路径被正确引用
for root, dirs, files in os.walk(TARGET_DIR):
    for file in files:
        # 跳过以 ._ 开头的文件
        if file.startswith('.'):
            print(f"Skipping file {os.path.join(root, file)} because it starts with '._'.")
            continue

        # 构建输出目录的路径
        relative_path = os.path.relpath(root, TARGET_DIR)
        output_dir = os.path.join(OUTPUT_DIR, relative_path)

        # 构建输出文件的完整路径
        output_file = os.path.join(output_dir, file)

        # 构建需要检查的auto目录路径
        auto_dir = os.path.join(OUTPUT_DIR, relative_path,
                                os.path.splitext(file)[0], os.path.splitext(file)[0], 'auto')

        # 判断该路径下是否已经完整提取
        print(f"Checking if directory {auto_dir} has been successfully processed...")

        # 使用 os.path.isdir 判断路径是否是目录，避免 os.path.abspath 可能的问题
        if os.path.isdir(auto_dir):
            if check_success(auto_dir):
                print(f"Directory {auto_dir} already processed successfully. Skipping...")
                continue
            else:
                print(
                    f"Directory {auto_dir} has not been processed yet or is incomplete. Proceeding with processing...")
        else:
            print(f"Directory {auto_dir} does not exist, proceeding with processing...")

        # 检查输出文件是否已经存在
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping...")
            continue

        # 检查该文件是否是PDF文件
        if file.endswith('.pdf'):
            # 如果是PDF文件，先检查所在文件夹是否已经处理过
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 检查该文件所在目录是否包含其他PDF文件
            pdf_count = len([f for f in os.listdir(root) if f.endswith('.pdf')])

            # 如果目录内有其他PDF文件，批量提取整个文件夹
            if pdf_count > 1:
                print(f"Processing folder {relative_path} as it contains multiple PDF files...")
                # 调用magic-pdf命令处理整个文件夹
                subprocess.run(['magic-pdf', '-p', root, '-o', output_dir, '-m', 'auto'])
            else:
                # 如果只有单个PDF文件，直接提取该文件
                print(f"Processing individual file {os.path.join(root, file)}...")
                # 调用magic-pdf命令处理单个PDF文件
                subprocess.run(['magic-pdf', '-p', os.path.join(root, file), '-o', output_dir, '-m', 'auto'])

        # 处理完成后，尝试清理CUDA缓存
        print("Cleaning CUDA cache")
        subprocess.run(['python', '-c', 'import torch; torch.cuda.empty_cache()'])

        # 检查是否有其他进程占用GPU，如果有，尝试结束它们
        print("Checking for other processes using GPU")
        result = subprocess.run(['nvidia-smi', '--id', str(GPU_ID), '-q'], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if 'python' in line:
                pid = line.split()[2]
                print(f"Killing process {pid} using GPU")
                subprocess.run(['nvidia-smi', '--id', str(GPU_ID), '--kill', pid])

        # 重置CUDA设备以释放显存
        print("Resetting GPU")
        reset_result = subprocess.run(['nvidia-smi', '--gpu-reset', '-i', str(GPU_ID)], capture_output=True, text=True)
        if reset_result.returncode == 0:
            print("GPU reset successfully.")
        else:
            print("Failed to reset GPU. Please check for other processes or restart.")

print("PDF files have been processed and output directories have been created as needed.")