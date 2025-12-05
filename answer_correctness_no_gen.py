import json
from langchain_ollama.llms import OllamaLLM
from langchain.schema import HumanMessage  # Ensure input is formatted correctly

# 简化的调试 Prompt
debug_prompt = '''
您是企业能碳知识核查专家，请严格按以下规则执行：

【评分规则】
| 维度       | 评分标准                                                                 | 
|------------|--------------------------------------------------------------------------|
| 语义完整度 | 覆盖参考答案核心要点数量比例（如数据来源、计算方法、关键指标）              | 
| 事实准确性 | 验证结果与真实数据一致性程度                                                | 
| 答案相似度 | 答案结构与术语规范程度（包括数据格式、专业术语、计算流程表述）               | 

【输出要求】
1. 输出格式要求如下,必须首先输出分数，并严格按照下列规范：
"semantic": 0-100整数, "factual": 0-100整数, "similarity": 0-100整数。
2. 若学生答案完全空白或无法识别，返回："semantic":0, "factual":0, "similarity":0。
3.可以对学生答案进行分析，并给出建议，分析和建议应在评分后给出。
4.输出应严格遵守以下流程：
    首先直接严格按照要求输出分数：
    "semantic": 0-100整数, "factual": 0-100整数, "similarity": 0-100整数。
    接着给出评分原因：
    **评分分析：**：
        **语义完整度**：
        **事实准确性**：
        **答案相似度**：
    最后给出教师建议：
    **教师建议：**：

请严格对照评分规则，给出专业客观的评估。
Correct Answer: {correct_answer}
Student Answer: {student_answer}
'''

# 输入文件路径
input_file = '/saves/DeepSeek-R1-1.5B-Distill/lora/eval_2025-03-04-13-48-20/generated_predictions.jsonl'

# 初始化 OllamaLLM 模型
llm1 = OllamaLLM(model="deepseek-r1:14b",
                options={"num_ctx": 8192, "temperature": 0}
                )

# 读取 JSONL 文件并逐行处理
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        # 加载单行 JSON 数据
        data = json.loads(line)
        correct_answer = data.get("label", "")  # 正确答案或问题
        student_answer = data.get("predict", "")  # 学生答案

        # 如果缺少必要字段，打印默认消息并跳过
        if not correct_answer or not student_answer:
            print("输入数据缺失")
            continue

        # 构建 Debug Prompt
        full_prompt = debug_prompt.format(correct_answer=correct_answer, student_answer=student_answer)

        # 将 prompt 封装为 HumanMessage
        input_message = [HumanMessage(content=full_prompt)]

        # 打印输入以调试
        print("=== 调试信息：输入 ===")
        print(full_prompt)

        # 调用 Ollama 进行评分
        try:
            response = llm1.invoke(input_message,options={"num_ctx": 8192, "temperature": 0})  # 传递 BaseMessages 类型

            # 打印输出以调试
            print("=== 调试信息：输出 ===")
            print(response)

        except Exception as e:
            print(f"调用失败: {str(e)}")
