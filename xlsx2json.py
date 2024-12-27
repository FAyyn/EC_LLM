import pandas as pd
import json

# 读取Excel文件
excel_file = '/workspace/DataProcess/测试问题（新）.xlsx'  # 请替换为你的Excel文件路径
df = pd.read_excel(excel_file)

# 假设问题列是"问题"，标准答案列是"标准答案"
questions_column = '问题'
answers_column = '标准答案'

# 创建一个列表来保存最终的JSON数据
json_data = []

# 遍历DataFrame中的每一行
for index, row in df.iterrows():
    entry = {
        'instruction': row[questions_column],
        'output': row[answers_column],
        'input': ''  # input字段为空
    }
    json_data.append(entry)

# 将数据写入JSON文件
output_json_file = '/workspace/DataProcess/bonito_output/eval.json'  # 请替换为你想保存的JSON文件路径
with open(output_json_file, 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

print(f"JSON文件已保存到 {output_json_file}")
