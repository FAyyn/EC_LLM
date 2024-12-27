import os
from charset_normalizer import from_path

def fix_encoding_save_as_new_file(original_file_path):
    try:
        # 检测文件编码
        matches = from_path(original_file_path)
        # 检查是否有匹配结果
        if not matches:
            raise ValueError("没有检测到任何编码匹配。")

        # 获取置信度最高的匹配项
        best_match = matches.best()
        if best_match is None:
            raise ValueError("没有找到最佳匹配项。")

        # 使用 best_match 对象的 output 方法来获取重新编码后的字节序列，并转换为字符串
        normalized_bytes = best_match.output()
        normalized_str = normalized_bytes.decode(best_match.encoding)

        # 获取文件的目录和文件名
        directory, filename = os.path.split(original_file_path)
        # 创建新文件名，例如：original_filename_fixed.md
        new_filename = f"{os.path.splitext(filename)[0]}_fixed.md"
        new_file_path = os.path.join(directory, new_filename)
        # 将修复后的字符串写入新文件
        with open(new_file_path, 'w', encoding='utf-8') as file:
            file.write(normalized_str)
        print(f"文件 {original_file_path} 编码修复成功，新文件保存为 {new_file_path}。")
    except Exception as e:
        print(f"文件 {original_file_path} 编码修复失败: {e}")

def process_markdown_files(directory):
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):  # 检查是否是Markdown文件
                file_path = os.path.join(root, file)
                print(f"正在处理文件：{file_path}")
                fix_encoding_save_as_new_file(file_path)

# 替换为你的Markdown文件所在的目录
directory_path = '/workspace/MinerU-DATA/5/1/auto'
process_markdown_files(directory_path)