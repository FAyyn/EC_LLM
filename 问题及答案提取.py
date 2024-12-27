import pandas as pd
import json
import os

def json_to_excel(json_file, excel_file, second_json_file=None, third_json_file=None, column_order=None):
    """
    将JSON数据提取并写入Excel文件，支持按列号选择字段的列位置。
    
    :param json_file: 输入的第一个JSON文件路径
    :param excel_file: 输出的Excel文件路径
    :param second_json_file: 可选的第二个JSON文件路径，用于提取Ec_OllamaN1.1回答
    :param third_json_file: 可选的第三个JSON文件路径，用于提取底座模型回答
    :param column_order: 可选的列号顺序，例如 [1, 2, 3] 或 [3, 1, 2]。
    """
    # 读取第一个JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取需要的字段
    rows = []
    for entry in data:
        row = {
            '题号': entry.get('id', ''),
            '问题': entry.get('instruction', ''),
            '标准答案': entry.get('output', '')  # 第一个文件的答案
        }
        rows.append(row)

    # 如果提供了第二个JSON文件，读取并提取数据
    if second_json_file:
        with open(second_json_file, 'r', encoding='utf-8') as f:
            second_data = json.load(f)

        # 将第二个JSON文件的回答提取并添加到现有数据中
        second_answers = {entry.get('id', ''): entry.get('output', '') for entry in second_data}

        # 为每一行添加来自第二个文件的回答
        for row in rows:
            row['Ec_OllamaN1.1回答'] = second_answers.get(row['题号'], '')  # 使用'题号'来匹配

    # 如果提供了第三个JSON文件，读取并提取数据
    if third_json_file:
        with open(third_json_file, 'r', encoding='utf-8') as f:
            third_data = json.load(f)

        # 将第三个JSON文件的回答提取并添加到现有数据中
        third_answers = {entry.get('id', ''): entry.get('output', '') for entry in third_data}

        # 为每一行添加来自第三个文件的回答
        for row in rows:
            row['底座模型回答'] = third_answers.get(row['题号'], '')  # 使用'题号'来匹配

    # 创建DataFrame
    df = pd.DataFrame(rows)

    # 如果提供了列号顺序，则根据列号调整列的顺序
    if column_order:
        # 确保列号在1到5之间（因为我们有5列）
        if all(1 <= col <= 5 for col in column_order):
            new_columns = ['题号', '问题', '标准答案', 'Ec_OllamaN1.1回答', '底座模型回答']
            df = df[new_columns].reindex(columns=[new_columns[i-1] for i in column_order])

    # 如果文件已存在，则读取并将新数据附加到已有数据后面
    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file, engine='openpyxl')

        # 合并现有数据和新数据
        df = pd.concat([existing_df, df], ignore_index=True)

    # 将数据写入Excel文件
    df.to_excel(excel_file, index=False, engine='openpyxl')

    print(f"数据已成功写入 {excel_file}")

# 使用示例
json_file = '/workspace/DataProcess/bonito_output/synthetic_qa_dataset_segmented_reshape_test_qwen.json'  # 替换为你的第一个JSON文件路径
excel_file = '/workspace/DataProcess/bonito_output/test问题带标准答案.xlsx'  # 输出的Excel文件路径
second_json_file = '/workspace/DataProcess/bonito_output/synthetic_qa_dataset_segmented_reshape_test_answer_EC_OllamaN1.1.json'  # 第二个JSON文件路径
third_json_file = '/workspace/DataProcess/bonito_output/synthetic_qa_dataset_segmented_reshape_test_base_model.json'  # 第三个JSON文件路径

# 用户输入列顺序 [1, 2, 3, 4, 5] 或其他顺序
# 例如将 "题号" 放在第2列，"问题" 放在第1列，"标准答案" 放在第3列，"Ec_OllamaN1.1回答" 放在第4列，"底座模型回答" 放在第5列
column_order = [1, 2, 3, 4, 5]  # 用户选择的列顺序，1表示题号列，2表示问题列，3表示标准答案列，4表示Ec_OllamaN1.1回答列，5表示底座模型回答列

json_to_excel(json_file, excel_file, second_json_file, third_json_file, column_order)
