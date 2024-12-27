import os
import torch
import shutil
from transformers import BertTokenizer, BertModel
from torch.nn.functional import log_softmax

# 初始化本地BERT模型和分词器
local_model_directory = "/workspace/chinese-bert"  # 本地模型目录路径
tokenizer = BertTokenizer.from_pretrained(local_model_directory)
model = BertModel.from_pretrained(local_model_directory)

def calculate_perplexity(text):
    # 对文本进行分词
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    # 获取模型的隐藏状态
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    # 获取logits
    logits = log_softmax(hidden_states[:, 0, :], dim=-1)  # 取第一个token的hidden state
    # 计算困惑度，确保logits没有多余的维度
    perplexity = torch.exp(torch.mean(logits[:, 1:].squeeze(dim=-1)))  # 排除[PAD] token
    return perplexity.item()

def ensure_directory_exists(path):
    # 如果目录不存在，则创建目录
    if not os.path.exists(path):
        os.makedirs(path)

def evaluate_markdown_files(source_directory, target_directory, threshold=30):
    # 确保目标目录存在
    ensure_directory_exists(target_directory)
    
    # 遍历源目录及其所有子目录中的所有文件
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            if filename.endswith('.md'):
                file_path = os.path.join(root, filename)
                print(f"评估文件：{file_path}")
                
                # 读取Markdown文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 计算困惑度
                perplexity = calculate_perplexity(content)
                print(f"文件 {filename} 的困惑度分数：{perplexity}")

                # 如果困惑度高于阈值，移动文件
                if perplexity > threshold:
                    relative_path = os.path.relpath(root, source_directory)
                    target_file_path = os.path.join(target_directory, relative_path, filename)
                    ensure_directory_exists(os.path.dirname(target_file_path))
                    shutil.move(file_path, target_file_path)
                    print(f"文件 {filename} 已移动到 {os.path.dirname(target_file_path)}")

# 设置源目录和目标目录
source_directory = "/workspace/MinerU-DATA"
target_directory = "/workspace/droped_file/困惑度"

# 执行评估
evaluate_markdown_files(source_directory, target_directory)