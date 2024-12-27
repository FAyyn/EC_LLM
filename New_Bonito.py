import json
import os
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from tqdm import tqdm
import torch
import gc

# 加载本地的Bonito模型
model_path = "/workspace/New Bonito"  # 替换为你的本地Bonito模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU
model = AutoModelForQuestionAnswering.from_pretrained(model_path).half().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def generate_qa_pairs(text):
    # 示例问题，可以根据需要自定义
    questions = [
        "这段文本的主要内容是什么？",
        "这段文本所涉及的电力领域知识有哪些？",
        "回答一些电力领域的问题："
    ]
    
    qa_pairs = []
    for question in questions:
        result = qa_pipeline({'question': question, 'context': text})
        qa_pairs.append({
            "question": question,
            "answer": result['answer']
        })
    return qa_pairs

def process_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    qa_data = {}
    if isinstance(data, list):
        for i, content in tqdm(enumerate(data), desc="Processing entries"):
            qa_data[f"entry_{i}"] = generate_qa_pairs(content)
            torch.cuda.empty_cache()  # 清理GPU缓存
            gc.collect()
    elif isinstance(data, dict):
        for filename, content in tqdm(data.items(), desc="Processing files"):
            qa_data[filename] = generate_qa_pairs(content)
            torch.cuda.empty_cache()  # 清理GPU缓存
            gc.collect()
    else:
        raise ValueError("Unsupported data format: must be a list or a dict")
        
    # 将生成的QA数据集保存为json格式
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(qa_data, json_file, ensure_ascii=False, indent=4)

# 示例用法
input_file = "/workspace/DataProcess/Data_Clean/1.md.json"  # 替换为之前生成的JSON文件路径
output_file = "/workspace/DataProcess/Data_Clean/qa_dataset.json"
process_json_file(input_file, output_file)
print(f"QA数据集已保存到 {output_file}")
