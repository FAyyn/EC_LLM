import json
import re
from typing import Dict  # 导入 Dict 类型
from langchain_ollama.llms import OllamaLLM
from langchain.schema import HumanMessage
import pandas as pd
from tqdm import tqdm
import os

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

def extract_scores(response: str) -> Dict[str, int]:
    """
    从模型响应中提取分数。
    返回格式：{"semantic": int, "factual": int, "similarity": int}
    """
    # 使用正则表达式匹配分数
    score_pattern = r'"semantic":\s*(\d+),\s*"factual":\s*(\d+),\s*"similarity":\s*(\d+)'
    match = re.search(score_pattern, response)
    
    if match:
        return {
            "semantic": int(match.group(1)),
            "factual": int(match.group(2)),
            "similarity": int(match.group(3)),
        }
    else:
        # 如果没有匹配到分数，返回默认值
        return {"semantic": 0, "factual": 0, "similarity": 0}

def process_file(input_file):
    # 初始化结果存储
    results = []
    scores = []
    
    # 初始化 OllamaLLM 模型
    llm1 = OllamaLLM(model="deepseek-r1:14b",
                    options={"num_ctx": 8192, "temperature": 0}
                    )

    # 读取并处理文件
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)
        
        for line in tqdm(f, total=total_lines, desc=f"处理文件 {os.path.basename(input_file)}"):
            data = json.loads(line)
            correct_answer = data.get("label", "")
            student_answer = data.get("predict", "")

            if not correct_answer or not student_answer:
                continue

            full_prompt = debug_prompt.format(
                correct_answer=correct_answer,
                student_answer=student_answer
            )

            try:
                response = llm1.invoke(
                    [HumanMessage(content=full_prompt)],
                    options={"num_ctx": 8192, "temperature": 0}
                )
                
                # 提取分数
                score = extract_scores(response)
                scores.append(score)

                # 保存完整结果
                results.append({
                    "correct_answer": correct_answer,
                    "student_answer": student_answer,
                    "analysis": response,
                    "scores": score
                })
                
            except Exception as e:
                print(f"处理异常: {str(e)}")
                continue

    # 保存JSON结果
    output_json = f"result_{os.path.basename(input_file).split('.')[0]}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 处理Excel数据
    if scores:
        df = pd.DataFrame(scores)
        
        # 计算统计指标
        stats = []
        for col in df.columns:
            avg = df[col].mean()
            pass_rate = (df[col] >= 60).mean() * 100
            excellent_rate = (df[col] >= 80).mean() * 100
            
            stats.append({
                "维度": col,
                "平均分": round(avg, 2),
                "合格率": f"{round(pass_rate, 2)}%",
                "优秀率": f"{round(excellent_rate, 2)}%"
            })
        
        # 保存到Excel
        output_excel = f"score_{os.path.basename(input_file).split('.')[0]}.xlsx"
        with pd.ExcelWriter(output_excel) as writer:
            df.to_excel(writer, sheet_name='原始分数', index=False)
            pd.DataFrame(stats).to_excel(writer, sheet_name='统计分析', index=False)
        
        print(f"\n处理完成！结果已保存到：\n{output_json}\n{output_excel}")

# 输入文件路径
input_file = '/saves/DeepSeek-R1-1.5B-Distill/lora/eval_2025-03-24-16-17-21_3867/generated_predictions.jsonl'

# 执行处理
process_file(input_file)