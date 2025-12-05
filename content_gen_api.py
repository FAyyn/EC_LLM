import os
import json
import time
from pathlib import Path
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# 导入 logging 模块
import logging

# 配置日志记录
logging.basicConfig(
    filename='/workspace/DataProcess/batch_job.log',  # 指定日志文件路径
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# 加载 embedding 模型，用于将 query 向量化
model_name = "/workspace/bge-small-zh"  # 请将此路径替换为您的本地模型路径
model_kwargs = {"device": "cuda:1"}     # 根据您的设备调整
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
    '/workspace/tlj/embedding/Energy_carbon2.faiss',  # 请将此路径替换为您的向量数据库路径
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 10})
logging.info("向量数据库加载完成。")

# 设置 OpenAI 客户端
# 请在环境变量中设置您的 API Key，避免将敏感信息直接写入代码
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    logging.error("请在环境变量中设置 DASHSCOPE_API_KEY。")
    raise ValueError("请在环境变量中设置 DASHSCOPE_API_KEY。")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=1800
)

# 设置模型接入点 ID
MODEL_ENDPOINT_ID = "deepseek-r1"  # 请在此处填写您的实际模型接入点 ID

# 定义系统提示和用户提示模板
system_prompt = """你的角色是作为助手，通过系统化的长思考过程深入探索问题，然后提供最终精确且准确的解决方案。"""

user_prompt_template = '''
基于以下上下文回答问题:

{context}

Question: {query}
'''

# 读取 JSON 文件中的问题
questions = []
json_file_path = '/workspace/DataProcess/bonito_output/carbon_questions_双碳工具.json'  # 请将此路径替换为您的 JSON 文件路径

# 判断 json_file_path 是否是有效的文件
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

# 限制每日处理的问题数量（根据 API 限制）
MAX_REQUESTS_PER_DAY = 200
questions_to_process = questions[:MAX_REQUESTS_PER_DAY]

# 设置输出文件路径
output_file_path = '/workspace/DataProcess/bonito_output/carbon_questions_双碳工具_answer.json'  # 请将此路径替换为您希望的输出文件路径

# 如果输出文件已存在，尝试读取已处理的问题数
if os.path.isfile(output_file_path):
    with open(output_file_path, 'r', encoding='utf-8') as outfile:
        processed_data = json.load(outfile)
        processed_questions = processed_data.get("questions", [])
        logging.info(f"已读取已处理的 {len(processed_questions)} 个问题。")
else:
    processed_questions = []
    logging.info("没有已处理的问题，开始新处理。")

# 计算已处理的问题数，继续处理剩余的问题
start_idx = len(processed_questions)

# 遍历问题列表，获取答案并保存到新字段
for idx, question in enumerate(questions_to_process[start_idx:], start=start_idx+1):
    query = question['content']
    logging.info(f"正在处理第 {idx}/{len(questions_to_process)} 个问题...")

    # 获取上下文，限制文档数量和长度
    context_documents = retriever.get_relevant_documents(query)[:5]  # 仅使用前 5 个最相关的文档
    context = "\n".join([doc.page_content[:2000] for doc in context_documents])  # 每个文档最多取前 2000 字符

    # 构建完整的提示
    user_prompt = user_prompt_template.format(context=context, query=query)
    # 注意，这里在 messages 中分别传递 system 和 user 的内容
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

    # 记录即将发送的请求内容
    logging.info("即将发送到 API 的请求内容：")
    logging.info(f"System Prompt: {system_prompt}")
    logging.info(f"User Prompt: {user_prompt}")

    try:
        # 调用方舟推理 API
        completion = client.chat.completions.create(
            model=MODEL_ENDPOINT_ID,
            messages=messages
        )

        # 处理响应数据
        if completion.choices and completion.choices[0].message:
            # 获取 message_content 对象
            message_content = completion.choices[0].message
            logging.info("已收到 API 响应。")
            
            # 从 message_content 对象中获取内容
            content = getattr(message_content, 'content', "")
            reasoning_content = getattr(message_content, 'reasoning_content', "")
            
            # 去除首尾空格
            final_answer = content.strip()
            reasoning_content = reasoning_content.strip()
            
            logging.info("思考过程：")
            logging.info(reasoning_content)
            logging.info("最终答案：")
            logging.info(final_answer)
            
            # 将答案、上下文和思考过程保存到问题字典中
            question['context'] = context
            question['answer'] = {
                'reasoning_content': reasoning_content,
                'final_answer': final_answer
            }
        else:
            logging.warning("API 返回的响应中没有有效的答案。")
            question['context'] = context
            question['answer'] = {
                'reasoning_content': "",
                'final_answer': "抱歉，未能生成有效答案。"
            }

    except Exception as e:
        logging.error(f"调用 API 出错: {str(e)}")
        question['context'] = context
        question['answer'] = {
            'reasoning_content': "",
            'final_answer': f"生成答案时出现错误：{str(e)}"
        }

    # 将本次处理的问题添加到已处理的问题列表中
    processed_questions.append(question)
    logging.info(f"已完成第 {idx} 个问题的处理。")

    # 在每个答案生成后，将当前已处理的问题列表保存到 JSON 文件
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump({"questions": processed_questions}, outfile, ensure_ascii=False, indent=4)
    logging.info(f"已将已处理的问题保存到 {output_file_path}")

logging.info("处理完成。")
