import json
import glob
import os
import re
from datasets import Dataset
from bonito import Bonito
from vllm.sampling_params import SamplingParams
from tqdm import tqdm
import torch
import torch.distributed as dist

# 设置手动选择GPU的部分
def select_gpu(gpu_id):
    print(f"Using GPU: {gpu_id}")
    torch.cuda.set_device(gpu_id)

def count_chinese_chars(text):
    # 使用正则表达式匹配中文字符
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    return len(chinese_char_pattern.findall(text))

def main(batch_size):
    # 手动选择GPU
    select_gpu(0)  # 假设我们选择GPU 0，你可以根据需要更改这个值

    # 读取 JSON 文件夹中的所有 JSON 文件
    json_dir_path = '/workspace/DataProcess/Segmented'  # 包含所有 JSON 文件的目录路径
    json_file_paths = []

    # 使用 os.walk 递归遍历所有子文件夹
    for root, dirs, files in os.walk(json_dir_path):
        for file in files:
            if file.endswith('_content_list_segmented.json'):
                json_file_paths.append(os.path.join(root, file))

    print(f"找到的 JSON 文件路径：{json_file_paths}")  # 打印匹配到的 JSON 文件路径，供调试使用

    # 初始化一个空的列表用于存储所有 JSON 文件的数据
    all_data = []

    for json_file_path in json_file_paths:
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                print(f"JSON 文件 {json_file_path} 读取成功")
                print(f"读取的数据样例：{data[:3]}")  # 打印前几个数据样例进行检查

                # 确保 'text' 列没有 None 值，并且中文字符大于等于30
                for item in data:
                    if 'text' in item and item['text'] and count_chinese_chars(item['text']) >= 30:
                        all_data.append(item)
                    else:
                        print(f"跳过无效条目: {item}")

        except json.JSONDecodeError as e:
            print(f"解析 JSON 文件 {json_file_path} 出错: {e}")

    # 如果合并后的数据为空，则退出程序
    if not all_data:
        print("所有 JSON 文件合并后为空，程序退出。")
        exit()

    print(f"合并后的数据样例：{all_data[:5]}")  # 打印前几个数据样例进行检查

    # 将列表数据转换为 Hugging Face Dataset 对象
    dataset = Dataset.from_list(all_data)

    # 初始化 Bonito 模型
    bonito = Bonito(
        "/workspace/Newbonito",  # 替换为 Bonito 模型的路径
        #gpu_memory_utilization=0.9,  # 增加GPU内存利用率
        #max_model_len=20576,
        tensor_parallel_size=4  # 假设你想要设置张量并行的大小为4
    )

    # 设置采样参数
    sampling_params = SamplingParams(max_tokens=2048, top_p=0.95, temperature=0.5, n=1)

    # 生成合成 QA 数据集，确保输出问题为中文，且取消评分问题
    synthetic_dataset = []
    for i in range(0, len(dataset), batch_size):
        # 获取批次数据并合并成一行文本
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        merged_text = " ".join(batch["text"])
        merged_batch = Dataset.from_dict({"text": [merged_text]})

        batch_synthetic = bonito.generate_tasks(
            merged_batch,
            context_col="text",  # JSON 文件中的列名
            task_type="qa",
            sampling_params=sampling_params,
            prompt_lang="zh",  # 设置输出问题语言为中文
            include_rating_questions=False  # 取消评分问题
        )
        synthetic_dataset.extend(batch_synthetic)
        torch.cuda.empty_cache()  # 每次处理完一个批次后清理缓存

    # 检查生成的合成数据集
    print("生成的合成 QA 数据集：")
    for item in synthetic_dataset[:5]:  # 只打印前5个数据样例进行检查
        print(item)

    # 将 Dataset 对象转换为可序列化的列表
    serializable_dataset = [dict(item) for item in synthetic_dataset]

    # 保存生成的数据集到一个固定的 JSON 文件
    output_file_path = "/workspace/DataProcess/bonito_output/synthetic_text_gen_dataset_segmented.json"  # 固定输出文件路径
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # 合并所有输出问题并保存为一个 JSON 文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_dataset, f, ensure_ascii=False, indent=4)

    print(f"合成 QA 数据集已保存到 {output_file_path}")

    # 确保在程序结束前销毁分布式进程组
    dist.destroy_process_group()
    print("分布式进程组已销毁")

if __name__ == "__main__":
    batch_size = 8  # 设置 batch size 为 32，你可以根据需要调整这个值
    main(batch_size)
