import json
import re
import torch
from transformers import MarianMTModel, MarianTokenizer  # Hugging Face 预训练翻译模型
from tqdm import tqdm  # 用于显示进度条

# 定义源语言和目标语言
from_code = "en"  # 英文
to_code = "zh"    # 中文

# 设置设备，优先使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载翻译模型和分词器
def load_model():
    model_name = f'Helsinki-NLP/opus-mt-{from_code}-{to_code}'
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# 加载模型
model, tokenizer = load_model()

# 定义提取函数
def extract_input_and_instruction(text):
    # 正则表达式提取`Article:`到`Question:`之间的内容作为input
    input_match = re.search(r"Article:(.*?)Question:", text, re.DOTALL)
    if input_match:
        input_text = input_match.group(1).strip()
    else:
        input_text = ""

    # 正则表达式提取`Question:`到`Answer:`之间的内容作为instruction
    instruction_match = re.search(r"Question:(.*?)Answer:", text, re.DOTALL)
    if instruction_match:
        instruction_text = instruction_match.group(1).strip()
    else:
        instruction_text = ""

    return input_text, instruction_text

# 定义翻译函数
def translate_text(text):
    try:
        # 使用tokenizer将文本转换为模型输入格式
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        # 进行翻译
        translated = model.generate(**inputs)
        # 解码翻译结果
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        print(f"翻译失败: {e}")
        return text  # 如果翻译失败，返回原始文本

# 读取目标json文件并处理
def process_json(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # 获取总条数
    total_entries = len(data)

    # 假设目标json结构有一个字段 "input" 需要处理
    output_data = []
    
    # 初始化编号
    index = 1

    # 使用 tqdm 显示进度条
    for entry in tqdm(data, desc="Processing entries", total=total_entries):
        if 'input' in entry:
            text = entry['input']
            input_text, instruction_text = extract_input_and_instruction(text)
            
            # 跳过没有instruction的条目
            if not instruction_text:
                continue
            
            # 将instruction构建为包含"阅读输入文本并回答问题后"和提取的instruction_text
            instruction = f"Read the following article and answer the question. {instruction_text}"
            
            # 翻译instruction
            translated_instruction = translate_text(instruction)
            
            # 构建输出字典
            new_entry = {
                "id": index,  # 为每个条目添加唯一编号
                "input": input_text,
                "instruction": translated_instruction,  # 使用翻译后的instruction
            }
            output_data.append(new_entry)
            index += 1
    
    # 将新的数据写入到输出json文件
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=4)

# 使用示例
input_filename = '/workspace/DataProcess/bonito_output/synthetic_qa_dataset_segmented.json'  # 目标json文件路径
output_filename = '/workspace/DataProcess/bonito_output/synthetic_qa_dataset_segmented_reshape.json'  # 输出json文件路径
process_json(input_filename, output_filename)

print(f"处理完成，结果已保存至 {output_filename}")
