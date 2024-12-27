import os
import json
from datasets import Dataset, DatasetDict

def load_json_to_hf_dataset(json_file_path):
    """
    加载 JSON 文件并转换为 Hugging Face 数据集格式
    :param json_file_path: JSON 文件路径
    :return: Hugging Face Dataset
    """
    # 读取 JSON 数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取 'text' 字段作为预训练文本
    texts = [entry["text"] for entry in data]

    # 构建 Hugging Face Dataset 格式
    dataset = Dataset.from_dict({"text": texts})
    
    return dataset

def save_to_disk(dataset, output_dir):
    """
    将 Hugging Face 数据集保存到磁盘
    :param dataset: Hugging Face 数据集
    :param output_dir: 保存路径
    """
    dataset.save_to_disk(output_dir)

def split_dataset(dataset, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    将数据集拆分为训练集、验证集和测试集
    :param dataset: Hugging Face 数据集
    :param train_size: 训练集占比
    :param val_size: 验证集占比
    :param test_size: 测试集占比
    :return: DatasetDict 格式的拆分数据集
    """
    # 使用 Hugging Face 数据集的 train_test_split 方法进行拆分
    dataset_split = dataset.train_test_split(test_size=1 - train_size)
    val_test_split = dataset_split["test"].train_test_split(test_size=test_size / (val_size + test_size))
    
    # 构建数据集字典
    dataset_dict = DatasetDict({
        "train": dataset_split["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"]
    })
    
    return dataset_dict

def merge_json_files_in_directory(directory_path):
    """
    遍历目标目录和所有子目录中的 JSON 文件，并将其内容合并为一个 JSON 对象。
    :param directory_path: 目标目录路径
    :return: 合并后的 JSON 数据
    """
    merged_data = []
    
    # 遍历目录及子目录
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # 只处理 JSON 文件
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                print(f"Loading JSON file: {json_file_path}")
                
                # 读取并合并数据
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    merged_data.extend(data)  # 将每个文件的数据合并到一个列表中
    
    return merged_data

def main():
    # 输入的目录路径，包含多个 JSON 文件
    directory_path = 'your_directory_path'  # 需要替换为你实际的目录路径
    
    # 合并目录中的所有 JSON 文件
    merged_data = merge_json_files_in_directory(directory_path)
    
    # 将合并后的数据保存为一个新的 JSON 文件
    output_json_file = 'merged_data.json'
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    print(f"Merged JSON data saved to {output_json_file}")
    
    # 读取合并后的 JSON 数据并转换为 Hugging Face 数据集格式
    dataset = Dataset.from_dict({"text": [entry["text"] for entry in merged_data]})
    
    # 可选：拆分数据集为训练集、验证集和测试集
    dataset_dict = split_dataset(dataset, train_size=0.8, val_size=0.1, test_size=0.1)
    
    # 保存数据集到磁盘（可选）
    save_to_disk(dataset_dict, 'output_dataset')  # 数据集保存路径

    # 打印数据集的基本信息
    print(dataset_dict)

if __name__ == '__main__':
    main()
