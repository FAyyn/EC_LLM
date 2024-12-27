import json
import glob
import os
from datasets import Dataset
from bonito import Bonito
from vllm.sampling_params import SamplingParams
from tqdm import tqdm

# 读取 JSON 文件夹中的所有 JSON 文件，包括子目录
json_dir_path = '/workspace/DataProcess/Data_Clean/exam'  # 包含所有 JSON 文件的目录路径，确保以 / 结尾
json_file_paths = []
for root, dirs, files in os.walk(json_dir_path):
    for file in files:
        if file.endswith('.json'):
            json_file_paths.append(os.path.join(root, file))

print(f"找到的 JSON 文件路径：{json_file_paths}")  # 打印匹配到的 JSON 文件路径，供调试使用

# 初始化一个空的列表用于存储所有 JSON 文件的数据
all_data = []

for json_file_path in tqdm(json_file_paths, desc="读取 JSON 文件"):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"JSON 文件 {json_file_path} 读取成功")
            print(f"读取的数据样例：{data[:3]}")  # 打印前几个数据样例进行检查

            # 确保 'text' 列没有 None 值
            for item in data:
                if 'text' in item and item['text']:
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
    "/workspace/New Bonito",  # 替换为 Bonito 模型的路径
    gpu_memory_utilization=0.9,
    max_model_len=20576
)

# 设置采样参数
sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=0.5, n=1)

# 生成合成 QA 数据集，确保输出问题为中文，且取消评分问题
synthetic_dataset = bonito.generate_tasks(
    dataset,
    context_col="text",  # JSON 文件中的列名
    task_type="exqa",
    sampling_params=sampling_params,
    prompt_lang="zh",  # 设置输出问题语言为中文
    include_rating_questions=False  # 取消评分问题
)

# 检查生成的合成数据集
print("生成的合成 QA 数据集：")
for item in synthetic_dataset:
    print(item)

# 将 Dataset 对象转换为可序列化的列表
serializable_dataset = [dict(item) for item in synthetic_dataset]

# 保存生成的数据集到一个固定的 JSON 文件
output_file_path = "/workspace/DataProcess/bonito_output/synthetic_qa_dataset1.json"  # 固定输出文件路径
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# 合并所有输出问题并保存为一个 JSON 文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(serializable_dataset, f, ensure_ascii=False, indent=4)

print(f"合成 QA 数据集已保存到 {output_file_path}")