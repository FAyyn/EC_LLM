import os
import shutil
import jieba
from collections import Counter, defaultdict
import math
import spacy
import re  # 导入正则表达式模块
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import spatial

# 加载spaCy中文模型
nlp = spacy.load("zh_core_web_trf")
# 加载英文模型
nlp_en = spacy.load("en_core_web_trf")

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten()
        }

def get_bert_embeddings(model, tokenizer, text, max_len=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dataset = TextDataset([text], tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=1)
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state
            embeddings.append(last_hidden_states[:, 0, :].cpu().numpy())  # 取CLS标记的嵌入

    return embeddings[0]

def remove_ambiguity_words(text, model, tokenizer, threshold=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    max_len = 512
    embeddings = get_bert_embeddings(model, tokenizer, text, max_len)

    # 计算词嵌入之间的相似度
    words = tokenizer.tokenize(text)
    similarity_matrix = np.zeros((len(words), len(words)))
    for i in range(len(words)):
        for j in range(len(words)):
            if i != j:
                # 确保索引不会超出范围
                if i < len(embeddings) and j < len(embeddings):
                    similarity_matrix[i, j] = 1 - spatial.distance.cosine(embeddings[i], embeddings[j])

    # 识别和去除歧义词汇
    ambiguous_words = []
    for i in range(len(words)):
        if np.any(similarity_matrix[i] > threshold):
            ambiguous_words.append(words[i])

    # 去除歧义词汇
    new_text = ' '.join([word for word in words if word not in ambiguous_words])
    return new_text

def load_stopwords(stopwords_path):
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f.readlines()])
        return stopwords
    except FileNotFoundError:
        print(f"停用词文件 {stopwords_path} 未找到。")
        return set()

def document_to_words(content, stopwords):
    words = jieba.cut(content.lower())
    return [word for word in words if word not in stopwords and word != ' ']

def remove_punctuation_and_normalize(content):
    # 使用spaCy进行分词
    doc = nlp(content)
    # 直接返回分词结果，不去除标点符号
    return " ".join([token.text for token in doc])

def remove_disambiguation_words(content, stopwords):
    # 使用spaCy进行无关词汇去除
    doc = nlp(content)
    return " ".join([token.text for token in doc if token.text not in stopwords])

def remove_latex_formulas(content):
    # 正则表达式匹配行内的LaTeX公式
    inline_formula_pattern = r'\$.*?\$'
    # 正则表达式匹配展示模式的LaTeX公式
    display_formula_pattern = r'\$\$.*?\$\$'
    
    # 删除行内的LaTeX公式
    content = re.sub(inline_formula_pattern, '', content, flags=re.DOTALL)
    # 删除展示模式的LaTeX公式
    content = re.sub(display_formula_pattern, '', content, flags=re.DOTALL)
    
    return content

def convert_to_lowercase(text):
    # 使用spaCy进行分词并将所有字母转换为小写
    doc = nlp_en(text)
    return " ".join([token.text.lower() for token in doc])

def ensure_directory_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"无法创建目录 {path}：{e}")
            raise

def process_markdown_files(source_directory, target_directory, chinese_stopwords_path, model_path, threshold=0.8):
    stopwords = load_stopwords(chinese_stopwords_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)

    ensure_directory_exists(target_directory)
    
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            if filename.endswith('.md'):
                file_path = os.path.join(root, filename)
                print(f"\n处理文件：{file_path}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    print(f"无法读取文件 {filename}：{e}")
                    continue
                
                # 去除LaTeX公式
                content = remove_latex_formulas(content)
                # 转换为小写
                content = convert_to_lowercase(content)
                # 分词
                content = remove_punctuation_and_normalize(content)
                # 去除无关词汇
                content = remove_disambiguation_words(content, stopwords)
                # 使用BERT模型去除歧义词汇
                new_content = remove_ambiguity_words(content, model, tokenizer, threshold)
                
                # 写入新文件
                target_file_path = os.path.join(target_directory, filename)
                try:
                    with open(target_file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"文件 {filename} 已处理并保存到 {target_file_path}")
                except Exception as e:
                    print(f"无法写入文件 {filename}：{e}")

# 设置源目录和目标目录
source_directory = "/workspace/MinerU-DATA/5/1/auto"
target_directory = "/workspace/MinerU-DATAclean"
chinese_stopwords_path = '/workspace/DataProcess/停用词表/stopwords/luanma.txt'
model_path = "/workspace/chinese-bert"

# 执行处理
process_markdown_files(source_directory, target_directory, chinese_stopwords_path, model_path)