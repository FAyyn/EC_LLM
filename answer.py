import json
import ollama
import time
from tqdm import tqdm  # 导入 tqdm 进度条

def process_json(input_file, output_file, model="qwen2.5:14b", max_length=2048):
    # 读取目标JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 使用 tqdm 处理每一项数据时显示进度条
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')  # 开始输出的JSON文件，以便后续添加数据

        for idx, entry in tqdm(enumerate(data), desc="Processing entries", unit="entry"):
            start_time = time.time()  # 记录开始时间

            # 获取 instruction 和 input
            instruction = entry.get("instruction", "")
            input_text = entry.get("input", "")
            entry_id = entry.get("id", None)  # 获取 id 字段

            # 拼接 instruction 和 input
            combined_input = f"{instruction}\n{input_text}"

            # 截断输入文本，确保其不超过最大长度
            if len(combined_input) > max_length:
                combined_input = combined_input[:max_length]

            # 准备消息格式，结合 instruction 和 input
            messages = [
                {"role": "user", "content": combined_input}
            ]

            # 调用 ollama 模型获取回答
            try:
                res = ollama.chat(
                    model=model,
                    stream=False,
                    messages=messages,
                    options={
                        "temperature": 0,
                        "num_predict":2048,
                        "top_p":0.9,
                        'stop': ['<EOT>'],
                    }
                )
                # res = ollama.generate(
                #         model=model, 
                #         prompt=combined_input,
                #         # suffix=suffix,
                #         options={
                #             'num_predict': 2048,
                #             'temperature': 0,
                #             'top_p': 0.9,  # 与 temperature 类似，较低的 top_p 值会让模型更加保守，较高的值会增加生成内容的多样性。
                #             'stop': ['<EOT>'],
                #         },
                #         )
 
                # 打印完整的 API 响应以调试
                print(f"Response for entry {entry_id}: {res}")

                # 获取并保存输出到 output 字段 (从返回结果中提取 response)
                output_text = res.get("message", {}).get("content", "")
                # output_text = response['response']
                if not output_text:
                    output_text = "No response or empty output."
            except Exception as e:
                print(f"Error processing entry {entry_id}: {e}")
                output_text = "Error"

            # 计算每个答案生成的时间
            elapsed_time = time.time() - start_time
            print(f"Processed entry {entry_id} in {elapsed_time:.2f} seconds.")

            # 构建保存的结果，包含 id、input 和 output
            result = {
                "id": entry_id,
                "instruction": instruction,
                "input": input_text,
                "output": output_text  # 保存从模型返回的 output
            }

            # 将结果写入文件，逐条写入，每个条目之间加逗号
            json.dump(result, f, ensure_ascii=False, indent=4)

            # 如果不是最后一项数据，写入逗号；否则结束 JSON 数组
            if idx < len(data) - 1:
                f.write(',\n')
            else:
                f.write('\n]')  # 结束 JSON 数组

# 示例使用
input_file = '/workspace/DataProcess/bonito_output/synthetic_qa_dataset_segmented_reshape.json'
output_file = '/workspace/DataProcess/bonito_output/synthetic_qa_dataset_segmented_reshape_qwen.json'
process_json(input_file, output_file)
