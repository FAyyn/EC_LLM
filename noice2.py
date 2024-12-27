import os
import shutil
import jieba
from collections import Counter, defaultdict
import math
import spacy
import re  # 导入正则表达式模块

# 加载spaCy中文模型
nlp = spacy.load("zh_core_web_trf")
# 加载英文模型
nlp_en = spacy.load("en_core_web_trf")

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

def process_markdown_files(source_directory, target_directory, chinese_stopwords_path):
    stopwords = load_stopwords(chinese_stopwords_path)
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
                # 去除歧无关词汇
                content = remove_disambiguation_words(content, stopwords)
                
                # 写入新文件
                target_file_path = os.path.join(target_directory, filename)
                try:
                    with open(target_file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"文件 {filename} 已处理并保存到 {target_file_path}")
                except Exception as e:
                    print(f"无法写入文件 {filename}：{e}")

# 设置源目录和目标目录
source_directory = ""
target_directory = ""
chinese_stopwords_path = '/workspace/DataProcess/停用词表/stopwords/luanma.txt'

# 执行处理
process_markdown_files(source_directory, target_directory, chinese_stopwords_path)