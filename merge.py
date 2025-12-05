import json

# 文件路径
file1 = '/workspace/DataProcess/bonito_output/carbon_questions_ESG体系_batch.jsonl'  # 第一个 JSONL 文件（例如，包含请求的 messages 部分）
file2 = '/workspace/DataProcess/bonito_output/carbon_questions_ESG体系_batch_answer.jsonl'  # 第二个 JSONL 文件（例如，包含响应的 messages 部分）
output_file = '/workspace/DataProcess/bonito_output/carbon_questions_ESG体系_batch_merge.jsonl'  # 输出文件，存储匹配后的 messages

# 加载第一个文件的数据
data_dict1 = {}
with open(file1, 'r', encoding='utf-8') as f1:
    for line in f1:
        data = json.loads(line.strip())
        custom_id = data.get('custom_id')
        if custom_id:
            # 仅提取 message 部分
            message = data.get('body', {}).get('messages')
            if message:
                data_dict1[custom_id] = message

# 加载第二个文件的数据并进行匹配
with open(file2, 'r', encoding='utf-8') as f2, \
     open(output_file, 'w', encoding='utf-8') as fout:
    for line in f2:
        data = json.loads(line.strip())
        custom_id = data.get('custom_id')
        if custom_id and custom_id in data_dict1:
            # 获取匹配的 message
            message1 = data_dict1[custom_id]
            # 提取第二个文件的 message 部分
            message2 = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message')
            if message2:
                # 合并两个 messages
                matched_data = {
                    'custom_id': custom_id,
                    'message1': message1,
                    'message2': message2
                }
                # 将合并后的数据写入输出文件
                fout.write(json.dumps(matched_data, ensure_ascii=False) + '\n')
        else:
            print(f"未找到匹配的 custom_id: {custom_id}")
