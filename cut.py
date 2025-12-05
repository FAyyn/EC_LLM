import json
import random

def random_sample_json_and_remove(input_file, output_file, remaining_file, sample_fraction=0.33):
    # 读取目标 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 计算抽取的数据量
    sample_size = int(len(data) * sample_fraction)

    # 随机抽取数据
    sampled_data = random.sample(data, sample_size)

    # 重新编号抽取的数据
    for idx, entry in enumerate(sampled_data):
        entry['id'] = idx + 1  # 重新编号从 1 开始

    # 保存抽取的数据到新的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

    # 从原数据中删除已抽取的条目
    remaining_data = [entry for entry in data if entry not in sampled_data]

    # 重新编号剩余的数据
    for idx, entry in enumerate(remaining_data):
        entry['id'] = idx + 1  # 重新编号从 1 开始

    # 保存剩余的数据到新的文件
    with open(remaining_file, 'w', encoding='utf-8') as f:
        json.dump(remaining_data, f, ensure_ascii=False, indent=4)

# 示例使用
input_file = '/workspace/wnp/carbon_questions_instruction_150.json'
output_file = '/workspace/wnp/carbon_questions_instruction_150_test.json'
remaining_file = '/workspace/wnp/carbon_questions_instruction_100.json'
random_sample_json_and_remove(input_file, output_file, remaining_file)
