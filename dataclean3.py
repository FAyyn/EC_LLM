import os
import re
import jieba
from collections import Counter, defaultdict
import math
import spacy
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import spatial
from charset_normalizer import from_path
from datasketch import MinHash
from nltk.stem import PorterStemmer
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


# 加载spaCy中文和英文模型
nlp = spacy.load("zh_core_web_trf")
nlp_en = spacy.load("en_core_web_trf")


# 从输入字符串中移除图片字符串 
def remove_image_string(input_string):
    pattern = r"!\[(.*?)\]\(.*?\)\{width=\".*?\" height=\".*?\"\}|!\[.*?\]\(.*?\)|\[.*?\]\{.*?\}"
    result = re.sub(pattern, "", input_string)
    return result

# 输入字符串中移除HTML字符串
def remove_html_string(input_string):
    pattern = r"<[^>]+>"
    result = re.sub(pattern, "", input_string)
    return result

# 输入字符串中移除CSS和JavaScript字符串
def remove_css_and_javascript(input_string):
    css_pattern = r"<style[^>]*>.*?</style>"
    input_string = re.sub(css_pattern, "", input_string, flags=re.DOTALL)
    
    javascript_pattern = r"<script[^>]*>.*?</script>"
    input_string = re.sub(javascript_pattern, "", input_string, flags=re.DOTALL)
    
    return input_string

# 转换为小写
def convert_to_lowercase(text):
    # 使用spaCy进行分词并将所有字母转换为小写
    doc = nlp_en(text)
    return " ".join([token.text.lower() for token in doc])

#删除latex公式
def remove_latex_formulas(input_string):
    # 正则表达式匹配行内的LaTeX公式
    inline_formula_pattern = r'\$.*?\$'
    
    # 正则表达式匹配展示模式的LaTeX公式
    display_formula_pattern = r'\$\$.*?\$\$|\\\[.*?\\\]'
    
    # 删除行内的LaTeX公式
    input_string = re.sub(inline_formula_pattern, '', input_string, flags=re.DOTALL)
    
    # 删除展示模式的LaTeX公式
    input_string = re.sub(display_formula_pattern, '', input_string, flags=re.DOTALL)
    
    return input_string

def detect_language_with_langdetect(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

# 进行预处理，并删除行首尾空格
def one_text_pre_process(text):
    text = remove_image_string(text)
    text = remove_html_string(text)
    text = remove_css_and_javascript(text)
    text = remove_latex_formulas(text)

    processed_text_split_lines = []
    for line in text.splitlines():
        if line.strip() in ["", ">"]:
            continue

        if detect_language_with_langdetect(line) == 'en':
            line = convert_to_lowercase(line)

        processed_text_split_lines.append(line)

    return "\n".join(processed_text_split_lines)



#进行文件编码修复和预处理，返回处理后的内容，不再保存为新文件
def fix_encoding_and_preprocess_file(original_file_path):
    try:
        matches = from_path(original_file_path)
        if not matches:
            raise ValueError("没有检测到任何编码匹配。")

        best_match = matches.best()
        if best_match is None:
            raise ValueError("没有找到最佳匹配项。")

        normalized_bytes = best_match.output()
        normalized_str = normalized_bytes.decode(best_match.encoding)

        processed_content = one_text_pre_process(normalized_str)

        return processed_content
    except Exception as e:
        print(f"文件 {original_file_path} 编码修复并预处理失败: {e}")
        return None


# TextDataset类
class TextDataset(torch.utils.data.Dataset):
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

# 获取BERT嵌入
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

# 去除歧义词汇
def remove_ambiguity_words(text, model, tokenizer, threshold=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #每个输入序列的最大token数
    max_len = 512
    # 获取BERT嵌入
    embeddings = get_bert_embeddings(model, tokenizer, text, max_len)
    
    # 只进行一次分词
    words = tokenizer.tokenize(text)
    
    # 计算词嵌入之间的相似度
    similarity_matrix = np.zeros((len(words), len(words)))
    for i in range(len(words)):
        for j in range(len(words)):
            if i!= j:
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

# 加载无关词
def load_stopwords(stopwords_path):
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f.readlines()])
        return stopwords
    except FileNotFoundError:
        print(f"停用词文件 {stopwords_path} 未找到。")
        return set()


def remove_punctuation_and_normalize(content, stopwords):
    # 将文本按空格分割成单词列表
    words = content.split()
    # 过滤掉无关词汇并返回结果
    return " ".join([word for word in words if word not in stopwords])

# 确保目录存在
def ensure_directory_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"无法创建目录 {path}：{e}")
            raise

# 定义批量处理Markdown文件函数，只进行编码修复和预处理
def batch_process_markdown_files(root_dir, output_root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(subdir, file)
                print(f"正在处理文件：{file_path}")
                fix_encoding_and_preprocess_file(file_path)

#进行段落去重
def text_to_words(text):
    """将中文文本分词成词的集合"""
    words = set(jieba.cut(text))
    return words
    #num_perm 哈希函数的数量 num_perm越高，算法精度越高
def calculate_minhash(text, num_perm=128):
    """计算文本的 MinHash 值"""
    words = text_to_words(text)
    m = MinHash(num_perm=int(num_perm))  # 确保num_perm是整数
    for word in words:
        m.update(word.encode('utf8'))
    return m

def read_markdown_file(file_path):
    """读取 Markdown 文件并返回所有文本内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def calculate_similarity(m1, m2):
    """计算两个 MinHash 对象的相似度"""
    return m1.jaccard(m2)

def remove_duplicates(markdown_content, num_perm=128, similarity_threshold=0.9):
    """使用 MinHash 算法去除 Markdown 文件中的重复内容"""
    lines = markdown_content.split('\n')
    minhash_dict = {}

    for line in lines:
        if line.strip():  # 忽略空行
            minhash = calculate_minhash(line, num_perm)
            minhash_key = tuple(minhash.digest())  # 将 NumPy 数组转换为元组
            if minhash_key not in minhash_dict:
                minhash_dict[minhash_key] = (line, minhash)  # 存储行文本和 MinHash 对象

    unique_lines = []
    seen_minhashes = set()

    for minhash_key, (line, minhash) in minhash_dict.items():
        if line not in unique_lines:
            unique_lines.append(line)
            seen_minhashes.add(minhash_key)

        for other_minhash_key, (other_line, other_minhash) in minhash_dict.items():
            if other_minhash_key not in seen_minhashes and calculate_similarity(minhash, other_minhash) > similarity_threshold:
                if other_line not in unique_lines:
                    unique_lines.append(other_line)
                    seen_minhashes.add(other_minhash_key)

    return '\n'.join(unique_lines)


def process_markdown_files(source_directory, target_directory, chinese_stopwords_path, model_path, num_perm=128, similarity_threshold=0.9):
    stopwords = load_stopwords(chinese_stopwords_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)

    ensure_directory_exists(target_directory)
    
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            if filename.endswith('.md'):
                file_path = os.path.join(root, filename)
                print(f"\n处理文件：{file_path}")
                
                # 编码修复和预处理，获取处理后的内容
                processed_content = fix_encoding_and_preprocess_file(file_path)
                if processed_content is None:
                    continue
                
                # 去除无关词
                processed_content = remove_punctuation_and_normalize(processed_content, stopwords)
                # 使用BERT模型去除歧义词汇
                processed_content = remove_ambiguity_words(processed_content, model, tokenizer, 0.8)
                
                # 使用MinHash去除重复内容
                unique_content = remove_duplicates(processed_content, num_perm, similarity_threshold)
                
                # 写入新文件
                target_file_path = os.path.join(target_directory, filename)
                try:
                    with open(target_file_path, 'w', encoding='utf-8') as f:
                        f.write(unique_content)
                    print(f"文件 {filename} 已处理并保存到 {target_file_path}")
                except Exception as e:
                    print(f"无法写入文件 {filename}：{e}")


#源目录和目标目录
source_directory = "/workspace/tlj/rag-retrieval-main/eval"
target_directory = "/workspace/tlj/rag-retrieval-main/MarkD.faiss"
chinese_stopwords_path = '/workspace/DataProcess/停用词表/stopwords/luanma'
model_path = "/workspace/bert-chinese"               

#处理
batch_process_markdown_files(source_directory, target_directory)
process_markdown_files(source_directory, target_directory, chinese_stopwords_path, model_path)