import os
import json
import concurrent.futures
from tqdm import tqdm
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
import torch.multiprocessing as mp
from itertools import cycle
import re

def init_model_on_device(device):
    return MT5ForConditionalGeneration.from_pretrained('/workspace/HeackMT5').to(device)

def clean_text(texts, model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def read_markdown_files(directory):
    markdown_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

def process_file(file_path, gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = init_model_on_device(device)
    tokenizer = MT5Tokenizer.from_pretrained('/workspace/HeackMT5', legacy=False)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配题目的开始
    pattern = r'(\d+\.)'  # 匹配数字后跟一个点
    questions = re.split(pattern, content)
    
    # 移除空字符串
    questions = [q for q in questions if q.strip()]
    
    cleaned_exams = []
    
    # 创建一个局部的tqdm进度条
    total_questions = len(questions) // 2
    pbar = tqdm(total=total_questions, desc=f"Processing {os.path.basename(file_path)}", position=0)
    
    # 重新组合分割的文本，确保每个题目从题号开始
    for i in range(0, len(questions), 2):
        if i + 1 < len(questions):  # 确保索引在范围内
            question_text = questions[i] + questions[i + 1].strip()
            cleaned_question = clean_text([question_text], model, tokenizer, device)[0]
            cleaned_exams.append({"exam": cleaned_question})
            pbar.update(1)
    
    pbar.close()
    return file_path, cleaned_exams

def process_files(input_directory, output_directory):
    markdown_files = read_markdown_files(input_directory)
    
    os.makedirs(output_directory, exist_ok=True)
    
    num_gpus = torch.cuda.device_count()
    gpu_ids = cycle(list(range(num_gpus)))
    total_files = len(markdown_files)
    
    # Create a main progress bar for all files
    pbar = tqdm(total=total_files, desc="Processing files", position=1)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for file_path in markdown_files:
            gpu_id = next(gpu_ids)  # Assign a GPU ID to each file
            futures.append(executor.submit(process_file, file_path, gpu_id))
        
        for future in concurrent.futures.as_completed(futures):
            file_path, cleaned_exams = future.result()
            pbar.update(1)
            
            # Create output JSON structure
            relative_file_path = os.path.relpath(file_path, input_directory)
            json_file_path = os.path.join(output_directory, os.path.splitext(relative_file_path)[0] + '.json')
            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(cleaned_exams, json_file, ensure_ascii=False, indent=4)

    pbar.close()

if __name__ == '__main__':
    input_directory_path = '/workspace/MinerU-DATA/3'
    output_directory_path = '/workspace/DataProcess/Data_Clean/exam'
    
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    process_files(input_directory_path, output_directory_path)