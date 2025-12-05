import json
from typing import Dict, List, Union
from langchain_ollama.llms import OllamaLLM

def load_data(json_path: str) -> Union[Dict, List]:
    """加载包含问题和标准答案的JSON文件"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded data type: {type(data)}")  # 打印数据类型
            print(f"Loaded data: {data}")  # 打印数据内容
            return data
    except Exception as e:
        print(f"加载JSON文件失败: {str(e)}")
        return None

def parse_predictions(response_text: str) -> List[Dict]:
    """解析模型输出的预测结果"""
    predictions = []
    for line in response_text.strip().split('\n'):
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            start = int(parts[0])
            end = int(parts[1])
            prob = float(parts[2])
            if end >= start:  # 过滤无效预测
                predictions.append({'start': start, 'end': end, 'probability': prob})
        except ValueError:
            continue
    
    # 按概率降序排序
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    return predictions

def get_predictions(context: str, question: str) -> List[Dict]:
    """通过Ollama调用模型获取预测"""
    prompt = (
        f"你是一个专注于能碳知识领域的专业助手，致力于为用户提供准确、专业且简练的回答。你的回答应基于能碳领域的专业知识，确保准确性，避免提供错误或误导性的信息。请注意，你的回答应限于能碳知识领域，避免回答超出此范围的问题。"
        f"上下文：{context}\n问题：{question}\n答案预测："
    )
    
    try:
        # 初始化OllamaLLM
        llm = OllamaLLM(
            model="deepseek-r1:14b",
            options={"num_ctx": 8192, "temperature": 0}
        )
        # 调用模型
        response = llm.invoke(prompt, options={"num_ctx": 8192, "temperature": 0})
        return parse_predictions(response)
    except Exception as e:
        print(f"模型调用失败: {str(e)}")
        return []

def evaluate_qa(json_path: str):
    """主评估函数"""
    data = load_data(json_path)
    if data is None:
        return
    
    # 检查数据格式
    if isinstance(data, dict):
        # 如果是字典，转换为列表形式
        data = [data]
    elif not isinstance(data, list):
        print("错误：JSON文件应包含一个字典或列表对象。")
        return
    
    total = len(data)
    sa_count = la_count = mrr_sum = 0

    for idx, item in enumerate(data, 1):
        try:
            # 提取上下文和问题
            context = item.get("input", "")
            question = "需要披露和改善的定量指标有哪些？"  # 根据问题调整
            gold = item.get("output", "")
            
            # 解析标准答案中的起始和结束位置
            gold_start = gold.find("<think>")  # 假设标准答案的起始位置
            gold_end = gold.find("</think>") + len("</think>")  # 假设标准答案的结束位置

            # 获取模型预测
            predictions = get_predictions(context, question)
            found_rank = None

            # 检查预测是否匹配标准答案
            for rank, pred in enumerate(predictions, 1):
                if pred['start'] <= gold_start and pred['end'] >= gold_end:
                    found_rank = rank
                    break

            # 计算指标
            if found_rank:
                if found_rank == 1:
                    sa_count += 1
                if found_rank <= 5:
                    la_count += 1
                mrr_sum += 1 / found_rank

            print(f"处理进度: {idx}/{total} | 当前MRR: {mrr_sum/idx:.4f}", end='\r')
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            continue

    print("\n\n评估结果:")
    print(f"严格准确率 (SaCC): {sa_count/total:.4f}")
    print(f"宽大准确率 (LaCC): {la_count/total:.4f}")
    print(f"平均倒数排名 (MRR): {mrr_sum/total:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python qa_eval.py <data.json>")
        sys.exit(1)
    
    evaluate_qa(sys.argv[1])