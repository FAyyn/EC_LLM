import json
import re
import spacy
import os
import jieba
from datasketch import MinHash
from transformers import BertTokenizer, BertForTokenClassification
import torch
from langdetect import detect, LangDetectException

# 加载 spaCy 的英文模型
nlp = spacy.load("en_core_web_trf")

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

def preprocess_text(text):
    # 删除行首尾和文本内容中的空格
    text = text.strip()
    # 移除图片字符串
    text = re.sub(r"!\[(.*?)\]\(.*?\)\{width=\".*?\" height=\".*?\"\}|!\[.*?\]\(.*?\)|\[.*?\]\{.*?\}",'', text)
    # 移除 HTML 标签
    text = re.sub(r'<.*?>', '', text)
    # 移除 CSS 和 JavaScript（假设在 <style> 和 <script> 标签内）
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
     # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    # 删除重复的三句内容
    text = remove_duplicates(text)
    # 过滤停用词
    text = filter_stopwords(text)
    
    # 检测文本是否为空
    if text.strip():  # 检查文本是否为空
        try:
            # 检测语言
            language = detect(text)
            if language == 'en':
                # 使用 spaCy 将所有字母转换为小写
                doc = nlp(text)
                text = " ".join([token.text.lower() for token in doc])
        except LangDetectException:
            print("Language detection failed, assuming English.")
    
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 处理每个 JSON 对象
    processed_data = []
    last_text_item = None  # 用于存储上一个text类型项的引用
    
    for item in data:
        if isinstance(item, dict):
            # 读取 text, table_caption, img_caption 字段
            text = item.get('text', '')
            table_caption = item.get('table_caption', '')
            img_caption = item.get('img_caption', '')

            # 如果当前项是text类型，则更新last_text_item
            if item.get('type') == 'text':
                last_text_item = item
                # 预处理当前text
                processed_text = preprocess_text(text)
                if processed_text:  # 只保留非空的text记录
                    last_text_item['processed_text'] = processed_text
                    processed_data.append(last_text_item)
            # 如果当前项是table或image类型，则将caption文本内容追加到上一个text字段的内容后
            elif item.get('type') in ['table', 'image'] and last_text_item:
                # 将caption列表转换为单个字符串
                if isinstance(table_caption, list):
                    table_caption = ' '.join(table_caption)
                if isinstance(img_caption, list):
                    img_caption = ' '.join(img_caption)
                
                # 追加caption到上一个text字段的内容后
                new_text = f"{last_text_item['text']} {table_caption} {img_caption}".strip()
                # 预处理新的文本
                processed_text = preprocess_text(new_text)
                if processed_text:  # 只保留非空的text记录
                    last_text_item['text'] = new_text
                    last_text_item['processed_text'] = processed_text
                    processed_data.append(last_text_item)
    
    return processed_data

def process_json_directory(directory_path):
    all_processed_data = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # 检查文件名是否包含 'content_list'
            if file.endswith('.json') and "content_list" in file:
                file_path = os.path.join(root, file)
                processed_data = process_json_file(file_path)
                
                # 将文件名中的 'content_list' 替换为 'content_processed'
                new_file_name = file.replace('content_list', 'content_processed')
                output_file_path = os.path.join(root, new_file_name)
                
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    json.dump(processed_data, file, ensure_ascii=False, indent=4)
                print(f"Processed file: {file_path}, saved to: {output_file_path}")
    
    return all_processed_data
# 指定要处理的目录路径
directory_path = '/workspace/ESG碳中和论文/碳中和论文'

# 处理所有 JSON 文件并汇总结果
all_results = process_json_directory(directory_path)