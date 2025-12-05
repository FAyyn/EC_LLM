import os
import json
import logging
from pathlib import Path
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# 配置日志记录
logging.basicConfig(
    filename='/workspace/DataProcess/batch_job.log',  # 日志文件路径
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# 加载嵌入模型，用于将查询向量化
model_name = "/workspace/bge-small-zh"  # 嵌入模型路径，请替换为你的实际路径
model_kwargs = {"device": "cuda:1"}     # 根据你的设备调整（如 "cpu" 或 "cuda:0"）
encode_kwargs = {"normalize_embeddings": True}

logging.info("正在加载嵌入模型...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
logging.info("嵌入模型加载完成。")

# 加载向量数据库
logging.info("正在加载向量数据库...")
vector_db = FAISS.load_local(
    '/workspace/tlj/embedding/Energy_carbon2.faiss',  # 向量数据库路径，请替换为你的实际路径
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 10})  # MMR 检索方式，初始取 10 个文档
logging.info("向量数据库加载完成。")

# 定义系统提示和用户提示模板
system_prompt = """你是一个专注于能碳知识领域的专业助手，致力于为用户提供准确、专业且简练的回答。你的回答应基于能碳领域的专业知识，确保准确性，避免提供错误或误导性的信息。请注意，你的回答应限于能碳知识领域，避免回答超出此范围的问题。"""

# 更新后的用户提示模板，包含 domain、category 和 sub_category
user_prompt_template = '''
基于以下上下文回答问题:

{context}

Question: 这是一个来自于{domain}领域的{category}相关问题，其原文是一份{sub_category}文件，{query}
'''

# 读取 JSON 文件中的问题
questions = []
json_file_path = '/workspace/DataProcess/bonito_output/carbon_questions_new_1.json'  # 输入 JSON 文件路径，请替换为你的实际路径

# 检查文件是否存在并读取
if os.path.isfile(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if 'questions' in data:
            questions.extend(data['questions'])
            logging.info(f"已加载 {len(questions)} 个问题。")
        else:
            logging.error(f"文件 {json_file_path} 中没有找到 'questions' 字段")
            exit(1)
else:
    logging.error(f"路径 {json_file_path} 不是有效的文件，请检查路径是否正确。")
    exit(1)

# 限制每日处理的问题数量
MAX_REQUESTS_PER_DAY = 100000  # 可根据需求调整
questions_to_process = questions[:MAX_REQUESTS_PER_DAY]

# 创建批处理输入的 JSONL 文件
input_jsonl_path = '/workspace/DataProcess/bonito_output/carbon_questions_new_batch_1.jsonl'  # 输出 JSONL 文件路径

with open(input_jsonl_path, 'w', encoding='utf-8') as outfile:
    for idx, question in enumerate(questions_to_process, 1):
        query = question['content']  # 获取问题内容
        
        # 提取分类信息
        classification = question['classification']
        domain = classification['domain']
        category = classification['category']
        sub_category = classification['sub_category']
        
        # 记录日志，便于调试
        logging.info(f"正在准备第 {idx}/{len(questions_to_process)} 个问题: {query}")
        logging.info(f"Domain: {domain}, Category: {category}, Sub_category: {sub_category}")
        
        # 获取上下文，限制文档数量为 8 个
        context_documents = retriever.get_relevant_documents(query)[:8]  # 取前 8 个相关文档
        context = "\n".join([doc.page_content[:2000] for doc in context_documents])  # 每个文档最多取 2000 字符
        
        # 构建用户提示，填充模板
        user_prompt = user_prompt_template.format(
            context=context,
            domain=domain,
            category=category,
            sub_category=sub_category,
            query=query
        )
        
        # 构建批处理请求数据
        request_data = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "deepseek-r1",  # 请替换为你的实际模型接入点 ID
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
        }
        
        # 将请求数据写入 JSONL 文件
        outfile.write(json.dumps(request_data, ensure_ascii=False) + '\n')

logging.info(f"已将所有请求写入批处理输入文件：{input_jsonl_path}")