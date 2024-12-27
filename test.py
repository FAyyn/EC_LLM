import json
import os
from datasets import Dataset
from tqdm import tqdm
import torch
from bonito import Bonito
from vllm import SamplingParams


def select_gpu(gpu_id=None):
    available_gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    if gpu_id is not None and f'cuda:{gpu_id}' in available_gpus:
        print(f"使用指定的 GPU: cuda:{gpu_id}")
        return torch.device(f'cuda:{gpu_id}')

    print("可用的 GPU 设备:")
    for i, gpu in enumerate(available_gpus):
        print(f"{i}: {gpu}")

    selected_gpu = int(input("请输入要使用的 GPU 编号: "))
    if selected_gpu in range(len(available_gpus)):
        return torch.device(available_gpus[selected_gpu])
    else:
        print("无效的 GPU 编号，默认使用 CPU。")
        return torch.device("cpu")


# 读取 JSON 文件夹中的所有 JSON 文件，包括子目录
json_dir_path = '/workspace/ESG碳中和论文/蓝碳论文/21世纪海上丝绸之路背景下的广东省蓝碳发展研究.pdf'  # 包含所有 JSON 文件的目录路径，确保以 / 结尾
json_file_paths = []
for root, dirs, files in os.walk(json_dir_path):
    for file in files:
        if file.endswith('_content_list.json'):
            json_file_paths.append(os.path.join(root, file))

print(f"找到的 JSON 文件路径：{json_file_paths}")

# 初始化一个空的列表用于存储所有 JSON 文件的数据
all_data = []

for json_file_path in tqdm(json_file_paths, desc="读取 JSON 文件"):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"JSON 文件 {json_file_path} 读取成功")
            print(f"读取的数据样例：{data[:3]}")

            for item in data:
                if 'text' in item and item['text']:
                    chinese_char_count = sum(1 for c in item['text'] if '\u4e00' <= c <= '\u9fff')
                    if chinese_char_count >= 20:
                        all_data.append(item)
                    else:
                        print(f"跳过中文字数小于 20 字的条目: {item}")
                else:
                    print(f"跳过无效条目: {item}")

    except json.JSONDecodeError as e:
        print(f"解析 JSON 文件 {json_file_path} 出错: {e}")

if not all_data:
    print("所有 JSON 文件合并后为空，程序退出。")
    exit()

print(f"合并后的数据样例：{all_data[:5]}")

# 将列表数据转换为 Hugging Face Dataset 对象
dataset = Dataset.from_list(all_data)

# 选择 GPU 设备
device = select_gpu(gpu_id='2')
print(f"使用设备：{device}")

# 清理 CUDA 缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_max_memory_allocated(device)
    torch.cuda.reset_max_memory_cached(device)


model_path = "/workspace/Newbonito"
# 初始化 Bonito 模型，并指定 CUDA 设备
model = Bonito(
    model_path,  # 替换为 Bonito 模型的实际路径
    max_model_len=65536,  # 减少最大序列长度
    gpu_memory_utilization=0.95  # 增加 GPU 内存利用率
)

# 生成合成 QA 数据集，确保输出问题为中文
synthetic_dataset = model.generate_tasks(
    dataset,  # 使用 Hugging Face Dataset 对象
    context_col="text",
    task_type="exqa",  # 指定任务类型为抽取式问答
    sampling_params=SamplingParams(max_tokens=256, top_p=0.95, temperature=0.5, n=1)
)

# 检查生成的合成数据集
print("生成的合成 QA 数据集：")
for item in synthetic_dataset[:5]:  # 只打印前5个样例
    print(item)

# 保存生成的数据集到一个固定的 JSON 文件
output_file_path = "/workspace/DataProcess/bonito_output/synthetic_qa_dataset1.json"
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# 合并所有输出问题并保存为一个 JSON 文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(synthetic_dataset, f, ensure_ascii=False, indent=4)

print(f"合成 QA 数据集已保存到 {output_file_path}")