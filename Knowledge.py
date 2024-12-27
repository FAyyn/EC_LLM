import os
import json

def extract_content_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        # 提取所有非空的 text 字段内容
        text_list = [item['text'] for item in data if 'text' in item and item['text']]
        
    return text_list

def main(root_folder):
    result_list = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('content_list.json'):
                file_path = os.path.join(root, file)
                
                # 提取文件名中除去 _content_list.json 的部分作为 input 内容
                input_content = file.replace('_content_list.json', '')
                
                # 提取文件中所有非空的 text 字段的内容
                text_list = extract_content_from_json(file_path)
                
                # 将 text 内容每5个分为一组
                for i in range(0, len(text_list), 5):
                    # 合并 text 内容到一行
                    output_content = ' '.join(text_list[i:i + 5])
                    
                    result = {
                        'instruction': "你是一个碳知识领域的科研工作者，以输入为题撰写相关论文",
                        'input': input_content,
                        'output': output_content
                    }
                    
                    result_list.append(result)
    
    # 将结果保存到一个新的 JSON 文件
    output_filename = '/workspace/DataProcess/bonito_output/knowledge.json'
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(result_list, output_file, ensure_ascii=False, indent=4)
        
    print(f'结果已保存到 {output_filename}')

if __name__ == "__main__":
    root_folder = '/workspace/ESG碳中和论文'  # 替换为你的根文件夹路径
    main(root_folder)
