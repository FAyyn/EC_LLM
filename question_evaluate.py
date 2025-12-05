import json
from langchain_ollama.llms import OllamaLLM
import time
import os

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_entry(entry, file_path, mode='a'):
    """保存单个条目到JSONL文件"""
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def save_response_log(response, log_file, entry_id, status):
    """保存原始response到日志文件"""
    log_entry = {
        "entry_id": entry_id,  # 问题序号
        "status": status,      # 通过或未通过
        "response": response    # 原始response
    }
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

def evaluate_question(prompt):
    """使用Ollama评估问题（最终修复版）"""
    evaluation_prompt = f"""
    请根据以下标准评估问题：
    1. 问题质量：问题是否专业且符合学术规范
    2. 上下文匹配：上下文信息能否支撑问题的回答
    3. 领域相关：问题是否严格属于能碳知识领域

    请严格按照以下JSON格式回应：
    {{
        "quality_pass": true/false,
        "context_match": true/false,
        "domain_relevance": true/false,
        "reason": "评估理由"
    }}

    待评估问题：
    {prompt}
    """

    llm = OllamaLLM(model="qwen2.5:14b",
                   options={"num_ctx": 8192, "temperature": 0})
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 直接调用模型并获取响应
            response = llm.invoke(evaluation_prompt,options={"num_ctx": 8192, "temperature": 0})
            
            # 调试信息：打印原始响应
            print(f"原始响应内容:\n{response}\n")
            
            # 尝试解析响应内容
            try:
                # 确保响应是字符串
                if isinstance(response, str):
                    # 去除可能的Markdown代码块标记
                    response = response.replace("```json", "").replace("```", "").strip()
                    return json.loads(response), response  # 返回解析后的JSON和原始response
                else:
                    raise ValueError("响应不是字符串格式")
            except json.JSONDecodeError as je:
                print(f"JSON解析失败: {je}")
                print(f"响应内容: {response}")
                return {
                    "quality_pass": False,
                    "context_match": False,
                    "domain_relevance": False,
                    "reason": "模型返回格式不符合要求"
                }, response  # 返回默认结果和原始response
                
        except Exception as e:
            print(f"评估出错（尝试 {attempt + 1}/{max_retries}）: {str(e)}")
            if attempt == max_retries - 1:
                return None, None
            time.sleep(1)  # 等待后重试

def process_entries(input_file, passed_file, failed_file, log_file):
    """处理JSONL条目"""
    entries = load_jsonl(input_file)
    passed_counter = 0
    failed_counter = 0
    
    # 清空输出文件和日志文件
    open(passed_file, 'w').close()
    open(failed_file, 'w').close()
    open(log_file, 'w').close()
    
    for idx, entry in enumerate(entries):
        try:
            # 提取用户问题
            user_content = next(
                msg["content"] for msg in entry["body"]["messages"] 
                if msg["role"] == "user"
            )
            
            # 执行评估
            evaluation, raw_response = evaluate_question(user_content)
            
            # 保存原始response到日志文件
            if raw_response:
                if evaluation and all([
                    evaluation["quality_pass"],
                    evaluation["context_match"],
                    evaluation["domain_relevance"]
                ]):
                    passed_counter += 1
                    entry["custom_id"] = f"request-{passed_counter}"
                    save_entry(entry, passed_file)
                    save_response_log(raw_response, log_file, f"passed-{passed_counter}", "passed")
                else:
                    failed_counter += 1
                    entry["custom_id"] = f"failed-{failed_counter}"
                    save_entry(entry, failed_file)
                    save_response_log(raw_response, log_file, f"failed-{failed_counter}", "failed")
            else:
                failed_counter += 1
                entry["custom_id"] = f"failed-{failed_counter}"
                save_entry(entry, failed_file)
                save_response_log("无响应", log_file, f"failed-{failed_counter}", "failed")
                
            print(f"已处理 {idx+1}/{len(entries)} 条，通过: {passed_counter} 条，未通过: {failed_counter} 条")
                
        except Exception as e:
            print(f"处理条目 {entry.get('custom_id','')} 时出错: {str(e)}")
            failed_counter += 1
            entry["custom_id"] = f"failed-{failed_counter}"
            save_entry(entry, failed_file)
            save_response_log("处理出错", log_file, f"failed-{failed_counter}", "failed")

if __name__ == "__main__":
    # 文件路径配置
    input_jsonl = "/workspace/DataProcess/Datasets/carbon_questions_new_batch_failed.jsonl"
    passed_jsonl = "/workspace/DataProcess/Datasets/carbon_questions_new_batch_failed_T.jsonl"
    failed_jsonl = "/workspace/DataProcess/Datasets/carbon_questions_new_batch_failed_F.jsonl"
    log_file = "/workspace/DataProcess/Datasets/carbon_questions_new_batch_log_2.jsonl"
    
    process_entries(input_jsonl, passed_jsonl, failed_jsonl, log_file)
    print("处理完成，有效问题已保存至", passed_jsonl)
    print("原始响应日志已保存至", log_file)