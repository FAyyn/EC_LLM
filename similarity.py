import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 模型路径
model_path = '/workspace/bert-chinese'  # 替换为你的模型路径

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# 读取Excel文件
excel_path = '/workspace/DataProcess/测试问题（新）.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(excel_path)

# 定义一个函数来计算余弦相似度
def calculate_cosine_similarity(text1, text2):
    # 确保输入是字符串
    if not isinstance(text1, str):
        text1 = str(text1)
    if not isinstance(text2, str):
        text2 = str(text2)

    # 将文本编码为模型的输入格式
    inputs = tokenizer(text1, text2, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # 将输入传递给模型
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state

    # 取最后一层的[CLS]标记的输出作为句子的表示
    sentence1_embedding = last_hidden_states[0, 0, :].detach().numpy().squeeze()
    sentence2_embedding = last_hidden_states[0, 1, :].detach().numpy().squeeze()

    # 计算两个句子嵌入的余弦相似度
    return cosine_similarity([sentence1_embedding], [sentence2_embedding])[0][0]

# 计算每一对文本的相似度并生成新的列
df['相似度_微调后回答_vs_答案'] = df.apply(lambda row: calculate_cosine_similarity(row['微调模型回答'], row['标准答案']), axis=1)
df['相似度_答案_vs_原模型回答'] = df.apply(lambda row: calculate_cosine_similarity(row['标准答案'], row['原模型回答']), axis=1)

# 保存到新的Excel文件
output_excel_path = '/workspace/DataProcess/相似度2.xlsx'  # 替换为你想要保存的新Excel文件路径
df.to_excel(output_excel_path, index=False)

print(f"相似度结果已保存到 {output_excel_path}")
