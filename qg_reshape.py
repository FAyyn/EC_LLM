import json

# 假设你的JSON文件名为data.json
input_filename = '/workspace/DataProcess/bonito_output/synthetic_qg_dataset_1.json'
output_filename = '/workspace/DataProcess/bonito_output/synthetic_qg_dataset_reshape.json'

# 读取JSON文件
with open(input_filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 初始化一个空列表来存储提取的数据
extracted_data = []

# 遍历JSON数据
for item in data:
    # 初始化提取的数据字典
    extracted_item = {}

    # 提取Article
    if 'input' in item and isinstance(item['input'], str):
        article_start = item['input'].find('Article:') + len('Article:')
        options_index = item['input'].find('Options:')
        article_end = options_index if options_index != -1 else len(item['input'])
        extracted_item['Article'] = item['input'][article_start:article_end].strip()

    # 提取Question
    if 'output' in item and isinstance(item['output'], str):
        question_index = item['output'].find('Question:')
        if question_index == -1:  # 如果output中没有Question字段
            extracted_item['Question'] = item['output'].strip()
        else:
            question_start = question_index + len('Question:')
            answer_index = item['output'].find('Answer:')
            if answer_index == -1:  # 如果output中没有Answer字段
                extracted_item['Question'] = item['output'][question_start:].strip()
            else:
                extracted_item['Question'] = item['output'][question_start:answer_index].strip()

    # 提取Options
    if 'input' in item and isinstance(item['input'], str) and 'Options:' in item['input']:
        options_start = item['input'].find('Options:') + len('Options:')
        answer_index = item['input'].find('Answer:')
        options_end = answer_index if answer_index != -1 else len(item['input'])
        extracted_item['Options'] = item['input'][options_start:options_end].strip()
    elif 'output' in item and isinstance(item['output'], str) and 'Options:' in item['output']:
        options_start = item['output'].find('Options:') + len('Options:')
        answer_index = item['output'].find('Answer:')
        options_end = answer_index if answer_index != -1 else len(item['output'])
        extracted_item['Options'] = item['output'][options_start:options_end].strip()

    # 提取Answer
    if 'input' in item and isinstance(item['input'], str) and 'Answer:' in item['input']:
        answer_start = item['input'].find('Answer:') + len('Answer:')
        question_index = item['input'].find('Question:')
        answer_end = question_index if question_index != -1 else len(item['input'])
        extracted_item['Answer'] = item['input'][answer_start:answer_end].strip()
    elif 'output' in item and isinstance(item['output'], str) and 'Answer:' in item['output']:
        answer_start = item['output'].find('Answer:') + len('Answer:')
        extracted_item['Answer'] = item['output'][answer_start:].strip()

    # 将提取的数据添加到列表中
    extracted_data.append(extracted_item)

# 将提取的数据写入新的JSON文件
with open(output_filename, 'w', encoding='utf-8') as file:
    json.dump(extracted_data, file, ensure_ascii=False, indent=4)

print(f'Extracted data has been written to {output_filename}')