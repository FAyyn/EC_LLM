import json

# 定义文件路径
input_file = '/workspace/DataProcess/bonito_output/carbon_questions_ESG体系_batch_merge.jsonl'  # 包含匹配后 message1 和 message2 的 JSONL 文件
output_json = '/workspace/DataProcess/bonito_output/carbon_questions_ESG体系_batch_instruction.json'  # 最终输出的 JSON 文件

# 存储最终的数据列表
final_data = []

# 读取输入文件并处理数据
with open(input_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        data = json.loads(line.strip())
        custom_id = data.get('custom_id')
        message1 = data.get('message1')  # 应该是一个包含多个字典的列表
        message2 = data.get('message2')  # 应该是一个字典

        # 初始化变量
        instruction = ''
        input_content = ''
        output = ''

        # 处理 message1，提取 instruction 和 input
        for msg in message1:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'system':
                instruction = content
            elif role == 'user':
                input_content = content

        # 处理 message2，提取 reasoning_content 和 content
        reasoning_content = message2.get('reasoning_content', '')
        assistant_content = message2.get('content', '')

        # 构建 output，按照指定格式
        output = f"<think>{reasoning_content}</think>\n<answer>{assistant_content}</answer>"

        # 构建最终的数据结构
        item = {
            'instruction': instruction,
            'input': input_content,
            'output': output
        }

        final_data.append(item)

# 将数据写入输出的 JSON 文件
with open(output_json, 'w', encoding='utf-8') as fout:
    json.dump(final_data, fout, ensure_ascii=False, indent=4)

print(f"数据已成功写入到 {output_json}")
