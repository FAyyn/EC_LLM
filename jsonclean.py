import json
import re
import spacy
import os
import jieba
from datasketch import MinHash
from transformers import BertTokenizer, BertForTokenClassification
import torch
from langdetect import detect

# 加载 spaCy 的英文模型
nlp=spacy.load("en_core_web_trf")

# 加载 BERT 模型和分词器
model_path = "/workspace/bert-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)

# 加载停用词表
chinese_stopwords_path = '/workspace/DataProcess/停用词表/stopwords/luanma'
with open(chinese_stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())

def text_to_words(text):
    """将中文文本分词成词的集合"""
    words = set(jieba.cut(text))
    return words

def calculate_minhash(text, num_perm=128):
    """计算文本的 MinHash 值"""
    words = text_to_words(text)
    m = MinHash(num_perm=int(num_perm))  # 确保num_perm是整数
    for word in words:
        m.update(word.encode('utf8'))
    return m

def calculate_similarity(m1, m2):
    """计算两个 MinHash 对象的相似度"""
    return m1.jaccard(m2)

def remove_duplicates(text, num_perm=128, similarity_threshold=0.9):
    """使用 MinHash 算法去除文本中的重复内容"""
    sentences = text.split('.')
    unique_segments = []
    minhashes = []
    
    # 将文本按三句分段
    segments = ['.'.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    
    for segment in segments:
        if segment.strip():  # 忽略空句子
            minhash = calculate_minhash(segment, num_perm)
            is_duplicate = False
            for existing_minhash in minhashes:
                if calculate_similarity(minhash, existing_minhash) > similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_segments.append(segment)
                minhashes.append(minhash)
    
    return '.'.join(unique_segments)

def filter_stopwords(text):
    """过滤停用词，并保持句子完整性"""
    words = jieba.lcut(text)
    filtered_words = [word for word in words if word not in stopwords]
    return ''.join(filtered_words)

def remove_ambiguous_words(text):
    """去除歧义词汇，并保持句子完整性"""
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    filtered_tokens = [token for token, pred in zip(tokens, predictions[0]) if pred.item() != 1]  # 假设 1 表示歧义词汇的标签
    return ''.join(filtered_tokens)

def preprocess_text(text):
    # 移除图片字符串
    text = re.sub(r"!\[(.*?)\]\(.*?\)\{width=\".*?\" height=\".*?\"\}|!\[.*?\]\(.*?\)|\[.*?\]\{.*?\}",'', text)
    # 移除 HTML 标签
    text = re.sub(r'<.*?>', '', text)
    # 移除 CSS 和 JavaScript（假设在 <style> 和 <script> 标签内）
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
    # 删除行首尾和文本内容中的空格
    text = text.strip()
    # 删除重复的三句内容
    text = remove_duplicates(text)
    # 过滤停用词
    text = filter_stopwords(text)
    # 去除歧义词汇
    #text = remove_ambiguous_words(text)
    
    # 检测文本是否为空
    if text.strip():  # 检查文本是否为空
        # 检测语言
        language = detect(text)
        
        if language == 'en':
            # 使用 spaCy 将所有字母转换为小写
            doc = nlp(text)
            text = " ".join([token.text.lower() for token in doc])
    
    return text

def process_json_file(file_path, output_directory):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 处理每个 JSON 对象
    processed_data = []
    for item in data:
        if isinstance(item, dict):
            text = item.get('text', '')
            processed_text = preprocess_text(text)
            if processed_text:  # 删除 text 内容为空的记录
                item['processed_text'] = processed_text
                processed_data.append(item)
        else:
            print(f"Unexpected item type: {type(item)}, value: {item}")
    
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 生成新的 JSON 文件
    output_file_path = os.path.join(output_directory, os.path.basename(file_path).replace('.json', '_processed.json'))
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, ensure_ascii=False, indent=4)
    
    return output_file_path

def process_json_directory(directory_path, output_directory):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json') and "_content_list" in file:
                file_path = os.path.join(root, file)
                new_file_path = process_json_file(file_path, output_directory)
                print(f"Processed file saved as: {new_file_path}")

# 指定要处理的目录路径和输出目录路径
directory_path = '/workspace/ESG碳中和论文/碳中和论文'
output_directory = '/workspace/Mineru-dataclean/'
process_json_directory(directory_path, output_directory)
