import os
import uuid
import pandas as pd
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

def load_keywords_from_excel(excel_path):
    """
    从Excel文件中加载中英文双碳领域关键词。
    假设Excel文件有两列，第一列为中文关键词，第二列为英文关键词。
    """
    keywords = []
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path, header=None)
        # 假设第二列是中文，第三列是英文
        keywords = df[0].dropna().tolist() + df[1].dropna().tolist()
        print(f"加载了 {len(keywords)} 个关键词。")
    except Exception as e:
        print(f"加载关键词时出错: {e}")
    return keywords

def get_chunk(folder_path, keywords):
    """
    遍历指定文件夹及其子文件夹下的所有Markdown文件，对文件内容进行分块处理，并根据子路径中的关键词设置元数据。
    """
    all_splits = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.md'):
                markdown_path = os.path.join(root, file_name)
                try:
                    with open(markdown_path, 'r', encoding='utf-8') as file:
                        markdown_content = file.read()
                    # 创建初始文档对象，包含'source'元数据
                    doc = Document(page_content=markdown_content, metadata={"source": markdown_path})
                    
                    # 查找双碳领域的关键词
                    found_keywords = [keyword for keyword in keywords if keyword in markdown_content]
                    if found_keywords:
                        doc.metadata["carbon_keywords"] = found_keywords  # 将匹配的关键词作为元数据存储
                    
                    # 分块处理
                    headers_to_split_on = [("#", "标题"), ("##", "标题"), ("###", "标题")]
                    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
                    rs = markdown_splitter.split_text(markdown_content)
                    
                    # 进一步切分，并保留'source'元数据
                    chunk_size = 500
                    chunk_overlap = 50
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    splits = text_splitter.split_documents(rs)
                    
                    # 更新每个分块的元数据，添加'source'和'carbon_keywords'
                    for split in splits:
                        split.metadata.update(doc.metadata)
                    
                    all_splits.extend(splits)
                except Exception as e:
                    print(f"处理文件 {file_name} 时出现错误: {e}")

    for doc in all_splits:
        # 获取文档所在的子路径
        sub_path = os.path.relpath(os.path.dirname(doc.metadata['source']), folder_path)
        if "碳达峰论文" in sub_path:
            doc.metadata.update({
                "esg_category": "相关论文",
                "sub_category": "碳达峰论文",               
            })
        elif "碳普惠论文" in sub_path:
            doc.metadata.update({
                "esg_category": "相关论文",
                "sub_category": "碳普惠论文",               
            })
        elif "碳中和论文" in sub_path:
            doc.metadata.update({
                "esg_category": "相关论文",
                "sub_category": "碳中和论文",
            })
        elif "碳足迹论文" in sub_path:
            doc.metadata.update({
                "esg_category": "相关论文",
                "sub_category": "碳足迹论文",
            })
        elif "ESG论文" in sub_path:
            doc.metadata.update({
                "esg_category": "相关论文",
                "sub_category": "ESG论文",
            })
        elif "ESG标准" in sub_path:
            doc.metadata.update({
                "esg_category": "标准和评级",
                "sub_category": "ESG标准",
            })
        elif "ESG评级" in sub_path:
            doc.metadata.update({
                "esg_category": "标准和评级",
                "sub_category": "ESG评级",
            })
        elif "企业白皮书和指南" in sub_path:
            doc.metadata.update({
                "esg_category": "企业白皮书和指南",
                "sub_category": "ESG企业白皮书和指南",
            })
        elif "上市企业报告" in sub_path:
            doc.metadata.update({
                "esg_category": "上市企业报告",
                "sub_category": "上市企业报告",
            })
        elif "ESG投资实践" in sub_path:
            doc.metadata.update({
                "esg_category": "案例和研究报告",
                "sub_category": "ESG投资实践",
            })
        elif "ESG研究报告" in sub_path:
            doc.metadata.update({
                "esg_category": "案例和研究报告",
                "sub_category": "ESG研究报告",
            })
        else:
            # 其他情况的元数据设置（如果有需要可以补充） 
            pass

    return all_splits


if __name__ == '__main__':
    folder_path = '/workspace/tlj/database/ESG体系'  # 替换为实际的文件夹路径
    excel_path = '/workspace/DataProcess/carbon_terms.xlsx'  # 替换为实际的Excel文件路径

    # 从Excel文件加载关键词
    keywords = load_keywords_from_excel(excel_path)

    if not keywords:
        print("没有加载到任何关键词，无法进行匹配。")
    else:
        all_splits = get_chunk(folder_path, keywords)
        if not all_splits:
            print("没有找到任何有效的Markdown文件内容，无法保存到向量数据库。")
        else:
            # 加载HuggingFaceBgeEmbeddings模型
            model_name = "BAAI/bge-small-zh"
            model_kwargs = {"device": "cuda:3"}
            encode_kwargs = {"normalize_embeddings": True}
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
            embedding_dim = 512
            try:
                # 创建FAISS索引
                index = faiss.IndexFlatL2(embedding_dim)
                # 创建向量存储
                vector_store = FAISS(
                    embedding_function=embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )

                uuids = [str(uuid.uuid4()) for _ in range(len(all_splits))]
                vector_store.add_documents(documents=all_splits, ids=uuids)

                # 保存向量存储到本地
                vector_store.save_local('/workspace/DataProcess/embedding/ESG.faiss')
                print('FAISS saved!')
            except Exception as e:
                print(f"保存向量数据库时出现错误: {e}")
