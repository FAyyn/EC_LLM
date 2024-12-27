import os
import shutil
import jieba
from collections import Counter, defaultdict
import math

# 从指定路径加载停用词
def load_stopwords(stopwords_path):
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f.readlines()])
        return stopwords
    except FileNotFoundError:
        print(f"停用词文件 {stopwords_path} 未找到。")
        return set()

# 计算词频（TF）
def calculate_tf(word, words):
    return words.count(word) / len(words)

# 计算逆文档频率（IDF）
def calculate_idf(word, documents):
    doc_count = sum(1 for doc in documents if word in doc)
    if doc_count == 0:
        return 0
    return math.log((1 + len(documents)) / (1 + doc_count)) + 1

# 计算TF-IDF值
def calculate_tf_idf(word, words, documents):
    tf = calculate_tf(word, words)
    idf = calculate_idf(word, documents)
    return tf * idf

# 将文档内容转换为单词列表，去除停用词和空格
def document_to_words(content, stopwords):
    words = jieba.cut(content.lower())
    return [word for word in words if word not in stopwords and word != ' ']

# 计算文档的TF-IDF值
def calculate_document_tf_idf(content, documents, stopwords):
    words = document_to_words(content, stopwords)
    tf_idf_values = {word: calculate_tf_idf(word, words, documents) for word in set(words)}
    return sum(tf_idf_values.values()), tf_idf_values

# 确保目录存在，如果不存在则创建
def ensure_directory_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"无法创建目录 {path}：{e}")
            raise

# 评估Markdown文件的TF-IDF密度，并根据阈值移动文件
def evaluate_markdown_files(source_directory, target_directory_low_density, chinese_stopwords_path, threshold=0.5):
    stopwords = load_stopwords(chinese_stopwords_path)
    ensure_directory_exists(target_directory_low_density)
    
    documents = []
    # 遍历源目录，读取所有Markdown文件内容，并进行分词
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            if filename.endswith('.md'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(document_to_words(content, stopwords))
    
    # 再次遍历源目录，对每个Markdown文件进行TF-IDF密度评估
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            if filename.endswith('.md'):
                file_path = os.path.join(root, filename)
                print(f"\n评估文件：{file_path}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    print(f"无法读取文件 {filename}：{e}")
                    continue
                
                # 计算当前文档的TF-IDF密度和各词的TF-IDF值
                tf_idf_density, tf_idf_values = calculate_document_tf_idf(content, documents, stopwords)
                print(f"文件 {filename} 的TF-IDF密度为 {tf_idf_density:.4f}")
                
                # 输出文件中TF-IDF值最高的前10个词
                features = sorted(tf_idf_values.items(), key=lambda item: item[1], reverse=True)
                print(f"文件 {filename} 的重要特征词TF-IDF值：{features[:10]}")

                # 根据TF-IDF密度决定是否移动文件
                if tf_idf_density > threshold:
                    print(f"文件 {filename} 的TF-IDF密度高于阈值 {threshold}，保留在原目录")
                else:
                    relative_path = os.path.relpath(file_path, source_directory)
                    target_file_path = os.path.join(target_directory_low_density, relative_path)
                    target_dir = os.path.dirname(target_file_path)
                    ensure_directory_exists(target_dir)
                    try:
                        shutil.move(file_path, target_file_path)
                        print(f"文件 {filename} 的TF-IDF密度低于阈值 {threshold}，已移动到 {target_directory_low_density}")
                    except Exception as e:
                        print(f"无法移动文件 {filename}：{e}")

# 设置源目录和目标目录
source_directory = "/workspace/MinerU-DATAclean"
target_directory_low_density = "/workspace/droped_file/文本密度"
chinese_stopwords_path = '/workspace/DataProcess/停用词表/stopwords/hit_stopwords.txt'

# 执行评估
evaluate_markdown_files(source_directory, target_directory_low_density, chinese_stopwords_path)