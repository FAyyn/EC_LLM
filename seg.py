import os
import json
import torch
import multiprocessing
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# 设置多进程启动方式为 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# 检查是否有GPU可用
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型和Tokenizer（BERT）
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)  # 将BERT模型移到GPU

# 目标目录和输出目录
target_directory = '/workspace/SRC/CCER'
output_directory = '/workspace/DataProcess/Segmented'

# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 获取目标目录中所有文件的数量
total_files = sum(len(files) for _, _, files in os.walk(target_directory)) - 1  # 减1是因为最后一个统计的是空文件夹

# 设置批量大小
batch_size = 64  # 可以根据GPU显存调整

# 全角转半角函数
def fullwidth_to_halfwidth(text):
    """
    将文本中的全角字符转换为半角字符
    """
    result = []
    for char in text:
        # 处理全角空格
        if ord(char) == 12288:  # 全角空格
            result.append(chr(32))  # 半角空格
        elif 65281 <= ord(char) <= 65374:  # 全角字符范围（包括全角标点、字母、数字等）
            result.append(chr(ord(char) - 0xFEE0))  # 转换为半角字符
        else:
            result.append(char)  # 非全角字符保持不变
    return ''.join(result)

# 计算句子的BERT嵌入
def get_sentence_embedding_batch(sentences):
    inputs = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)  # 将输入张量移到GPU
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # 使用[CLS] token的嵌入作为句子的嵌入
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # 将结果移回CPU以进行进一步处理
    return embeddings

# 计算两个句子之间的余弦相似度
def cosine_sim(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)[0][0]

# 合并相似的句子，并确保每个段落不超过2048个字
def merge_sentences(sentences, similarity_threshold=0.6, max_length=2048):
    merged_paragraphs = []
    current_paragraph = [] 
    current_paragraph_length = 0  # 当前段落的字数

    for i, sentence in enumerate(sentences):
        # 计算当前句子的字数
        sentence_length = len(sentence)

        if current_paragraph:
            # 计算当前段落最后一句话与当前句子的相似度
            last_embedding = get_sentence_embedding_batch([current_paragraph[-1]])
            current_embedding = get_sentence_embedding_batch([sentence])
            similarity = cosine_sim(last_embedding, current_embedding)

            # 如果相似度高于阈值并且字数不会超过限制，将句子合并到当前段落
            if similarity > similarity_threshold and current_paragraph_length + sentence_length <= max_length:
                current_paragraph.append(sentence)
                current_paragraph_length += sentence_length
            else:
                # 否则，保存当前段落并开始一个新段落
                merged_paragraphs.append(''.join(current_paragraph))
                current_paragraph = [sentence]
                current_paragraph_length = sentence_length
        else:
            current_paragraph.append(sentence)
            current_paragraph_length = sentence_length

    # 处理最后一个段落
    if current_paragraph:
        merged_paragraphs.append(''.join(current_paragraph))

    return merged_paragraphs

# 处理单个文件的函数
def process_file(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有文本内容（每一行作为一个句子），并跳过空行
    sentences = []
    for doc in data:
        if doc.get('type') == 'text' and doc.get('text'):
            sentences.extend([sentence.strip() for sentence in doc['text'].split('。') if sentence.strip()])

    # 合并句子为段落
    merged_paragraphs = merge_sentences(sentences)

    # 将全角字符转换为半角字符
    merged_paragraphs = [fullwidth_to_halfwidth(paragraph) for paragraph in merged_paragraphs]

    return merged_paragraphs  # 返回处理后的段落（没有编号）

# 使用并行处理多个文件
def process_files_in_parallel():
    # 获取所有文件路径
    file_paths = []
    for root, dirs, files in os.walk(target_directory):
        for file in files:
            if file.endswith('_content_list.json'):
                file_paths.append(os.path.join(root, file))

    all_paragraphs = []  # 用来存储所有文件处理后的段落
    total_files_count = len(file_paths)
    processed_files_count = 0  # 已处理的文件数量

    # 使用ThreadPoolExecutor并行处理文件
    with ThreadPoolExecutor() as executor:
        for result in executor.map(process_file, file_paths):
            all_paragraphs.extend(result)  # 合并每个文件的段落
            processed_files_count += 1  # 每处理一个文件，已处理的文件数加1
            
            # 显示整体的完成度
            completion_percentage = (processed_files_count / total_files_count) * 100
            print(f"文件处理进度: {processed_files_count}/{total_files_count} ({completion_percentage:.2f}%)")

    # 为所有段落统一编号
    all_segmented_data = []
    for new_index, text in enumerate(all_paragraphs):
        all_segmented_data.append({'text': text.strip(), 'index': new_index})

    # 将所有合并的段落输出到一个JSON文件
    output_file_path = os.path.join(output_directory, 'CCER.json')
    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        json.dump(all_segmented_data, out_f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    process_files_in_parallel()
    print("处理完成。")
