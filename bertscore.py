import pandas as pd
from bert_score import BERTScorer

# 读取Excel文件
def read_excel(file_path, sheet_name=0):
    return pd.read_excel(file_path, sheet_name=sheet_name)

# 初始化BERTScorer
def init_bertscorer(model_type, device='cuda:3'):
    # 使用预定义的模型类型初始化BERTScorer
    scorer = BERTScorer(model_type=model_type, device=device)
    return scorer

# 计算BERTScore
def calculate_bertscore(df, standard_column, compare_column, scorer):
    # 确保所有值都是字符串
    standard_answers = df[standard_column].astype(str).tolist()
    compare_answers = df[compare_column].astype(str).tolist()
    
    # 计算BERTScore
    P, R, F1 = scorer.score(compare_answers, standard_answers)
    return P, R, F1

# 主函数
def main():z 
    # Excel文件路径
    file_path = '/workspace/DataProcess/bonito_output/test问题带标准答案.xlsx'  # 替换为你的Excel文件路径
    output_file_path = '/workspace/DataProcess/BERTScore_ECllamaN1.1结果.xlsx'  # 输出文件路径
    
    # 读取Excel文件
    df = read_excel(file_path)
    
    # 预定义的模型类型
    model_type = 'bert-base-chinese'  # 使用预定义的模型类型
    # 初始化BERTScorer
    scorer = init_bertscorer(model_type)
    
    # 计算微调模型回答的BERTScore
    standard_column = '标准答案'  # 标准答案列名
    original_column = '底座模型回答'    # 原模型回答列名
    P_original, R_original, F1_original = calculate_bertscore(df, standard_column, original_column, scorer)
    
    # 计算原模型回答的BERTScore
    fine_tuned_column = 'Ec_OllamaN1.1回答'    # 微调后答案列名
    P_fine_tuned, R_fine_tuned, F1_fine_tuned = calculate_bertscore(df, standard_column, fine_tuned_column, scorer)
    
    # 将结果添加到数据框中
    df['BERTScore_P_original'] = P_original.tolist()
    df['BERTScore_R_original'] = R_original.tolist()
    df['BERTScore_F1_original'] = F1_original.tolist()
    df['BERTScore_P_SFT'] = P_fine_tuned.tolist()
    df['BERTScore_R_SFT'] = R_fine_tuned.tolist()
    df['BERTScore_F1_SFT'] = F1_fine_tuned.tolist()
    
    # 输出结果到新的Excel文件
    df.to_excel(output_file_path, index=False)
    print(f"结果已保存到 {output_file_path}")

if __name__ == "__main__":
    main() 
