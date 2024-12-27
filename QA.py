import json

# 假设你的原始 JSON 文件名为 original_json_file.json
input_filename = '/workspace/DataProcess/bonito_output/synthetic_qg_dataset_reshape.json'
output_filename = '/workspace/DataProcess/bonito_output/QA.json'

# 读取原始 JSON 文件
with open(input_filename, 'r', encoding='utf-8') as file:
    raw_json_list = json.load(file)

# 检查列表是否为空
if not raw_json_list:
    print("The JSON list is empty.")
else:
    # 创建一个空列表来存储所有新的 JSON 结构
    new_json_list = []

    # 遍历列表中的每个字典
    for raw_json in raw_json_list:
        # 使用 get 方法安全地访问字典中的键，如果键不存在则返回 None 或指定的默认值
        article = raw_json.get("Article", "")
        question = raw_json.get("Question", "")
        options = raw_json.get("Options", "")
        answer = raw_json.get("Answer", "")

        # 创建新的 JSON 结构
        new_json = {
            "instruction": "阅读文本内容并回答问题",
            "input": f"Article: {article}\nOptions: {options}\nQuestion: {question}",
            "output": answer
        }
        # 将新的 JSON 结构添加到列表中
        new_json_list.append(new_json)

    # 将新的 JSON 列表保存到新的文件
    with open(output_filename, 'w', encoding='utf-8') as file:
        json.dump(new_json_list, file, indent=4, ensure_ascii=False)

    print(f'New JSON file saved as {output_filename}')
