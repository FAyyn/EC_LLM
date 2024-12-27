import os
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, set_seed
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vllm import SamplingParams
from awq import AutoAWQForCausalLM 

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable device-side assertions
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

class QuantizedBonito:
    def __init__(self, model_name_or_path, rank, world_size, device="cuda", gpu_memory_utilization=0.9, max_model_len=20576):
        setup_distributed(rank, world_size)
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.model = AutoAWQForCausalLM.from_quantized(
            model_name_or_path, fuse_layers=True
        ).to(self.device)
        self.model = DDP(self.model, device_ids=[rank])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

    def generate_task(self, input_text: str, task_type: str, sampling_params: dict) -> dict:
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=self.max_model_len).to(self.device)
        try:
            outputs = self.model.module.generate(input_ids, do_sample=True, **sampling_params)
        except RuntimeError as e:
            print(f"Error during generation: {e}")
            return {"input": input_text, "output": "Generation failed."}
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"input": input_text, "output": output_text}

def main(rank, world_size):
    # 配置路径和模型
    target_directory = "/workspace/ESG碳中和论文"  # 目标目录
    output_file = "/workspace/DataProcess/bonito_output"  # 输出文件路径
    model_name_or_path = "/workspace/bonito-v1-awq"

    # 初始化模型
    bonito = QuantizedBonito(model_name_or_path, rank, world_size, gpu_memory_utilization=0.9, max_model_len=20576)

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

    cleanup_distributed()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)