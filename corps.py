import os
import json
import re

# 目标目录和输出文件路径
target_directory = '/workspace/DataProcess/Segmented'  # 请根据实际情况修改路径
output_file = '/workspace/DataProcess/bonito_output/CP.json'  # 输出的文件路径

# 判断字符串中是否包含中文字符
def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fa5]', text))

# 过滤中文字符大于30字的text
def filter_text_by_length(text, min_length=30):
    # 只提取中文字符
    chinese_text = ''.join(re.findall('[\u4e00-\u9fa5]', text))
    return len(chinese_text) >= min_length

# 处理目标目录中的所有JSON文件
def process_json_files():
    all_filtered_texts = []
    
    # 遍历目标目录中的所有文件
    for root, dirs, files in os.walk(target_directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # 读取每个JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        
                        # 提取所有含有text字段的内容
                        for doc in data:
                            # 检查doc是否包含'文本'字段
                            if isinstance(doc, dict) and 'text' in doc:
                                text = doc['text'].strip()
                                if filter_text_by_length(text):
                                    all_filtered_texts.append(text)
                            # 如果'内容'嵌套在更深层次，处理嵌套结构
                            elif isinstance(doc, dict):
                                # 递归查找text字段
                                def extract_text_from_dict(d):
                                    if isinstance(d, dict):
                                        for key, value in d.items():
                                            if key == 'text' and isinstance(value, str):
                                                if filter_text_by_length(value):
                                                    all_filtered_texts.append(value)
                                            elif isinstance(value, dict):
                                                extract_text_from_dict(value)
                                            elif isinstance(value, list):
                                                for item in value:
                                                    extract_text_from_dict(item)
                                extract_text_from_dict(doc)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")
    
    # 按顺序编号
    numbered_texts = [{'index': idx + 1, 'text': text} for idx, text in enumerate(all_filtered_texts)]
    
    # 将结果保存到新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(numbered_texts, out_f, ensure_ascii=False, indent=4)
    print(f"处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    process_json_files()
