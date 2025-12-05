#!/bin/bash

# 检查是否提供了目标目录和输出目录
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <target_directory> <output_directory> [<GPU_ID>]"
    exit 1
fi

# 目标目录和输出目录
TARGET_DIR="$1"
OUTPUT_DIR="$2"

# 检查是否提供了GPU ID，默认使用 GPU 0
GPU_ID=${3:-0}

# 确保目标目录和输出目录是绝对路径
TARGET_DIR=$(realpath "$TARGET_DIR")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

# 检查输出目录是否存在，如果不存在则创建
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# 打印选择的GPU ID
echo "Using GPU ID: $GPU_ID"

# 设置CUDA_VISIBLE_DEVICES环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 检查输出目录是否包含完整的提取结果
check_success() {
    local dir="$1"
    
    # 确保路径中的空格和特殊字符不会干扰命令
    dir=$(realpath "$dir")

    # 检查该目录下是否存在至少三个 .pdf 文件
    pdf_count=$(find "$dir" -type f -name '*.pdf' | wc -l)
    if [ "$pdf_count" -lt 3 ]; then
        echo "Missing PDF files in $dir"
        return 1
    fi

    # 检查该目录下是否有至少三个 .json 文件
    json_count=$(find "$dir" -type f -name '*.json' | wc -l)
    if [ "$json_count" -lt 3 ]; then
        echo "Missing JSON files in $dir"
        return 1
    fi

    return 0
}

# 遍历目标目录下的所有文件
# 使用 find 命令，并确保路径被正确引用
find "$TARGET_DIR" -type f | while read -r file; do
    # 跳过以 ._ 开头的文件
    if [[ "$(basename "$file")" == ._* ]]; then
        echo "Skipping file $file because it starts with '._'."
        continue
    fi

    # 提取文件的相对路径
    relative_path=$(dirname "$file")
    relative_path=${relative_path#"$TARGET_DIR/"}

    # 构建输出目录的路径
    output_dir="$OUTPUT_DIR/$relative_path"

    # 提取文件名（不含路径）
    filename=$(basename "$file")
    # 构建输出文件的完整路径
    output_file="$output_dir/$filename"

    # 构建需要检查的auto目录路径
    auto_dir="$OUTPUT_DIR/$relative_path/$(basename "$file" .pdf).pdf/$(basename "$file" .pdf)/auto"

    # 判断该路径下是否已经完整提取
    echo "Checking if directory $auto_dir has been successfully processed..."

    # 使用 -d 判断路径是否是目录，避免 realpath 可能的问题
    if [ -d "$auto_dir" ]; then
        if check_success "$auto_dir"; then
            echo "Directory $auto_dir already processed successfully. Skipping..."
            continue
        else
            echo "Directory $auto_dir has not been processed yet or is incomplete. Proceeding with processing..."
        fi
    else
        echo "Directory $auto_dir does not exist, proceeding with processing..."
    fi

    # 检查输出文件是否已经存在
    if [ -f "$output_file" ]; then
        echo "File $output_file already exists. Skipping..."
        continue
    fi

    # 检查该文件是否是PDF文件
    if [[ "$file" == *.pdf ]]; then
        # 如果是PDF文件，先检查所在文件夹是否已经处理过
        if [ ! -d "$output_dir" ]; then
            mkdir -p "$output_dir"
        fi

        # 检查该文件所在目录是否包含其他PDF文件
        pdf_count=$(find "$relative_path" -type f -name '*.pdf' | wc -l)
        
        # 如果目录内有其他PDF文件，批量提取整个文件夹
        if [ "$pdf_count" -gt 1 ]; then
            echo "Processing folder $relative_path as it contains multiple PDF files..."
            output_dir="$OUTPUT_DIR/$relative_path"
            
            # 调用magic-pdf命令处理整个文件夹
            magic-pdf -p "$relative_path" -o "$output_dir" -m auto
        else
            # 如果只有单个PDF文件，直接提取该文件
            echo "Processing individual file $file..."
            
            # 调用magic-pdf命令处理单个PDF文件
            magic-pdf -p "$file" -o "$output_file" -m auto
        fi
    fi

    # 处理完成后，尝试清理CUDA缓存
    echo "Cleaning CUDA cache"
    python -c "import torch; torch.cuda.empty_cache()"

    # 检查是否有其他进程占用GPU，如果有，尝试结束它们
    echo "Checking for other processes using GPU"
    nvidia-smi --id $GPU_ID | grep 'python' | awk '{print $3}' | xargs -I {} nvidia-smi --id $GPU_ID --kill $3

    # 重置CUDA设备以释放显存
    echo "Resetting GPU"
    if nvidia-smi --gpu-reset -i $GPU_ID; then
        echo "GPU reset successfully."
    else
        echo "Failed to reset GPU. Please check for other processes or restart."
    fi

done

echo "PDF files have been processed and output directories have been created as needed."
