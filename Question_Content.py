import json

# 假设你的输入 JSON 文件名为 input.json，输出文件名为 output.json
input_filename = '/workspace/DataProcess/bonito_output/synthetic_qg_dataset_1.json'
output_filename = '/workspace/DataProcess/bonito_output/synthetic_question_with_content.json'

# 读取输入文件
with open(input_filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 创建一个空列表来存储新的数据
new_data = []

# 遍历输入数据中的每个条目
for item in data:
    # 提取问题
    question = item['output']
    
    # 提取 Article 到 Options 之间的内容
    content_start = item['input'].find('Article:') + len('Article:')
    content_end = item['input'].find('Options:')
    content = item['input'][content_start:content_end].strip()

    # 创建新的条目
    new_item = {
        'question': question,
        'content': content
    }

    # 将新的条目添加到列表中
    new_data.append(new_item)

# 将新的数据写入输出文件
with open(output_filename, 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)

print(f'New JSON file has been created: {output_filename}')
