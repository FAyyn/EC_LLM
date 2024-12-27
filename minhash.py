from datasketch import MinHash
import jieba
from collections import defaultdict
import os

def text_to_words(text):
    """将中文文本分词成词的集合"""
    words = set(jieba.cut(text))
    return words

def calculate_minhash(text, num_perm=128):
    """计算文本的 MinHash 值"""
    words = text_to_words(text)
    m = MinHash(num_perm=num_perm)
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

def process_markdown_files(source_directory, target_directory, num_perm=128, similarity_threshold=0.9):
    """批量处理目录中的所有 Markdown 文件"""
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)  # 如果目标目录不存在，则创建目录

    for filename in os.listdir(source_directory):
        if filename.endswith(".md"):
            file_path = os.path.join(source_directory, filename)
            markdown_content = read_markdown_file(file_path)
            unique_markdown_content = remove_duplicates(markdown_content, num_perm, similarity_threshold)
            
            # 保存去重后的内容到新目录
            new_filename = f"unique_{filename}"
            new_file_path = os.path.join(target_directory, new_filename)
            with open(new_file_path, 'w', encoding='utf-8') as file:
                file.write(unique_markdown_content)
            print(f"Processed {filename}, saved as {new_filename}")

#源目录和目标目录
source_directory = "/workspace/MinerU-DATA/5/1/auto"
target_directory = "/workspace/MinerU-DATAclean"

process_markdown_files(source_directory,target_directory)