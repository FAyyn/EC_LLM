import os
import json
import time
from operator import itemgetter
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import HumanMessage

# 加载 embedding 模型，用于将 query 向量化
model_name = "/workspace/bge-small-zh"  # 请将此路径替换为您的本地模型路径
model_kwargs = {"device": "cuda:1"}  # 根据您的设备调整
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 加载向量数据库
vector_db = FAISS.load_local(
    '/workspace/tlj/embedding/Energy_carbon2.faiss',  # 请将此路径替换为您的向量数据库路径
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# 初始化本地模型
llm = OllamaLLM(model="qwen2.5:14b")

# 修改后的 Prompt 模板
system_prompt = """你的角色是作为助手，通过系统化的长思考过程深入探索问题，然后提供最终精确且准确的解决方案。这需要通过分析、总结、探索、重新评估、反思、回溯和迭代的全面循环，来发展经过深思熟虑的思考过程。请将你的回答结构化为两个主要部分：思考（Thought）和解决方案（Solution）。在思考部分，详细描述你的推理过程，使用指定的格式：
<|begin_of_thought|>{{思考内容，步骤之间用 '\\n\\n' 分隔}}<|end_of_thought|>
每个步骤应包括详细的内容，例如分析问题、总结相关发现、提出新想法、验证当前步骤的准确性、修正错误以及重新审视之前的步骤。在解决方案部分，基于思考部分的各种尝试、探索和反思，系统地呈现你认为正确的最终解决方案。解决方案应保持逻辑清晰、准确简洁的表达风格，并详细说明得出结论所需的必要步骤，格式如下：
<|begin_of_solution|>{{最终格式化、精确且清晰的解决方案}}<|end_of_solution|>
现在，请按照上述指南尝试解决以下问题："""

user_prompt = HumanMessagePromptTemplate.from_template('''
基于以下上下文回答问题:

{context}

Question: {query}
''')
full_chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    user_prompt
])

# 设置 chat chain
chat_chain = {
    "context": itemgetter("context"),
    "query": itemgetter("query"),
    "chat_history": itemgetter("chat_history"),
} | full_chat_prompt | llm

# 读取 JSON 文件中的问题
questions = []
json_file_path = '/workspace/DataProcess/bonito_output/carbon_questions_ESG体系.json'  # 请将此路径替换为您的 JSON 文件路径

# 判断 json_file_path 是否是有效的文件
if os.path.isfile(json_file_path):
    # 读取单个 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if 'questions' in data:
            questions.extend(data['questions'])
        else:
            print(f"文件 {json_file_path} 中没有找到 'questions' 字段")
            exit(1)
else:
    print(f"路径 {json_file_path} 不是有效的文件，请检查路径是否正确。")
    exit(1)

# 初始化 chat_history
chat_history = []

# （可选）限制每日处理的问题数量
MAX_REQUESTS_PER_DAY = 100  # 您可以根据需要调整
questions_to_process = questions[:MAX_REQUESTS_PER_DAY]

total_questions = len(questions_to_process)

# 遍历问题列表，获取答案并保存到新字段
for idx, question in enumerate(questions_to_process, 1):
    query = question['content']
    progress_percentage = (idx / total_questions) * 100
    print(f"正在处理第 {idx}/{total_questions} 个问题... ({progress_percentage:.2f}%)")

    # 获取上下文，限制文档数量和长度
    context_documents = retriever.invoke(query)[:2]  # 仅使用前 2 个最相关的文档
    context = "\n".join([doc.page_content[:500] for doc in context_documents])  # 每个文档最多取前 500 字符

    # 准备输入
    inputs = {
        'context': context,
        'query': query,
        'chat_history': chat_history
    }

    # 在传递信息到模型前打印所有需要传递的信息
    print("传递给模型的输入：")
    print(json.dumps(inputs, ensure_ascii=False, indent=4))

    # 调用本地模型
    try:
        answer = chat_chain.invoke(inputs)
    except Exception as e:
        print(f"调用本地模型出错: {e}")
        answer = "抱歉，生成答案时出现错误。"

    # 将答案保存到问题中
    question['RAG+chunk+mmr-local'] = answer.strip()

    # 更新 chat_history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(HumanMessage(content=answer))

    # 每处理完一个问题，等待 2 秒钟再处理下一个问题
    time.sleep(2)

print("所有问题处理完毕，正在保存结果...")

# 将修改后的问题列表保存到新的 JSON 文件
output_file_path = '/workspace/DataProcess/bonito_output/carbon_questions_ESG体系_answer.json'  # 请将此路径替换为您希望的输出文件路径
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump({"questions": questions_to_process}, outfile, ensure_ascii=False, indent=4)

print("Processing completed.")
