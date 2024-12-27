import os
import shutil

# 从指定路径加载停用词
def load_stopwords(stopwords_path):
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            # 读取文件并去除每行的空白字符，然后转换成集合
            stopwords = set([line.strip() for line in f.readlines()])
        return stopwords
    except FileNotFoundError:
        print(f"停用词文件 {stopwords_path} 未找到。")
        return set()

# 检查文本是否包含乱码
def is_garbled(text, stopwords, threshold=0.1):
    # 定义允许的字符集合，包括停用词中的字符和一些基本符号
    allowed_chars = stopwords.union(set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n"))
    
    # 计算不在允许字符集合中的字符数量
    garbled_count = sum(1 for char in text if char not in allowed_chars)
    
    # 计算乱码字符占总字符的比例
    garbled_ratio = garbled_count / len(text) if text else 0
    
    return garbled_ratio

# 检查并移动Markdown文件
def check_and_move_md_files(source_dir, target_dir, stopwords_path, threshold=0.2):
    # 加载停用词
    stopwords = load_stopwords(stopwords_path)
    
    # 如果目标目录不存在，则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 遍历源目录中的所有文件
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith('.md'):
                file_path = os.path.join(root, filename)
                print(f"正在检查文件：{file_path}")
                
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # 计算乱码率
                garbled_ratio = is_garbled(content, stopwords, threshold)
                print(f"文件 {filename} 的乱码率为 {garbled_ratio:.4f}")
                
                # 如果乱码率超过阈值，则移动文件
                if garbled_ratio > threshold:
                    relative_path = os.path.relpath(root, source_dir)
                    target_file_dir = os.path.join(target_dir, relative_path)
                    if not os.path.exists(target_file_dir):
                        os.makedirs(target_file_dir)
                    target_file_path = os.path.join(target_file_dir, filename)
                    shutil.move(file_path, target_file_path)
                    print(f"文件 {filename} 已移动到 {target_file_dir}")

# 定义源目录和目标目录路径
source_directory = "/workspace/MinerU-DATA"
target_directory = "/workspace/droped_file/乱码"
stopwords_path = '/workspace/DataProcess/停用词表/stopwords/hit_stopwords.txt'

# 执行检查和移动文件的操作
check_and_move_md_files(source_directory, target_directory, stopwords_path)