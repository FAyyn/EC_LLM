import os
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
from bonito import AbstractBonito

# 设置 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 根据实际可用 GPU 修改
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用设备端断言

# 配置内存分配器
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native,garbage_collection_threshold:0.8,max_split_size_mb:128,expandable_segments:True'

class BonitoModel(AbstractBonito):
    def __init__(self, model_name_or_path, device="cuda", gpu_memory_utilization=0.9, max_model_len=20576):
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        
        # 使用 DataParallel 实现多卡并行
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for parallel inference.")
            self.model = torch.nn.DataParallel(self.model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def generate_task(self, input_text: str, task_type: str, sampling_params: dict) -> dict:
        # 限制输入的最大序列长度
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=self.max_model_len).to(self.device)
        try:
            outputs = self.model.generate(input_ids, do_sample=True, **sampling_params)
        except RuntimeError as e:
            print(f"Error during generation: {e}")
            return {"input": input_text, "output": "Generation failed."}
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"input": input_text, "output": output_text}

# 配置 cuBLAS 和 cuFFT 环境变量
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2:16:8'  # 默认配置
torch.backends.cuda.cufft_plan_cache.max_size = 4096  # 设置 cuFFT 计划缓存的最大容量

# 配置路径和模型
target_directory = "/workspace/ESG碳中和论文"  # 目标目录
output_file = "/workspace/DataProcess/bonito_output"  # 输出文件路径
# 使用更小的模型，例如distilbert
model_name_or_path = "/workspace/bonito-v1"  # 替换为较小的模型

# 初始化模型
device = "cuda:3" if torch.cuda.is_available() else "cpu"
bonito = BonitoModel(model_name_or_path, device, gpu_memory_utilization=0.8, max_model_len=20000)

# 设置随机种子
set_seed(42)

# 采样参数
sampling_params = {
    "max_new_tokens": 128,  # 缩短生成的最大序列长度
    "top_p": 0.95,
    "temperature": 0.7,
    "num_return_sequences": 1
}

# 分批处理函数
def process_batch(data, bonito, sampling_params):
    batch_instructions = []
    for item in data:
        text = item.get("text", "")  # 假设内容在 `text` 字段中
        if not text.strip():
            continue
        instruction = bonito.generate_task(text, task_type="nli", sampling_params=sampling_params)
        batch_instructions.append(instruction)
    # 每次处理完一个批次后清理缓存
    torch.cuda.empty_cache()
    torch._C._cuda_clearCublasWorkspaces()
    return batch_instructions

# 遍历文件夹
all_instructions = []
for root, _, files in os.walk(target_directory):
    for file in files:
        if file.endswith("content_list.json"):
            file_path = Path(root) / file
            print(f"Processing: {file_path}")

            # 读取 JSON 文件
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 分批处理数据
            batch_size = 1  # 根据需要调整批量大小
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_instructions = process_batch(batch, bonito, sampling_params)
                all_instructions.extend(batch_instructions)

# 保存为 HuggingFace 格式 JSON 文件
hf_dataset = DatasetDict({"train": Dataset.from_list(all_instructions)})
hf_dataset.save_to_disk(output_file)

print(f"Instructions saved to {output_file}")