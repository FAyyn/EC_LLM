"""
能碳术语自动分类系统 v1.4

更新内容：
1. 修复tqdm兼容性问题
2. 优化进度显示方式
3. 增强版本适配性
"""

import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm  # 修改导入方式
from typing import List

# ================== 分类配置 ==================
CLASS_FEATURES = {
    "核心参数": {
        "cn": ["因子", "率", "系数", "强度", "当量"],
        "en": ["Factor", "Rate", "EF", "CI"]
    },
    "计量方法": {
        "cn": ["计算", "核算", "评估", "监测"],
        "en": ["Calculation", "Monitoring", "MRV"]
    },
    "项目类型": {
        "cn": ["项目", "机制", "交易", "信用"],
        "en": ["Project", "Programme", "CCER"]
    }
}

# ================== 核心函数 ==================
def auto_classify_term(term: str) -> str:
    """稳健的分类函数"""
    if pd.isna(term):
        return "未分类"
    
    term = str(term).strip()
    term_lower = term.lower()
    
    # 中文特征匹配
    for category, features in CLASS_FEATURES.items():
        if any(kw in term for kw in features["cn"]):
            return category
    
    # 英文特征匹配
    for category, features in CLASS_FEATURES.items():
        if any(re.search(rf"\b{kw.lower()}\b", term_lower) for kw in features["en"]):
            return category
    
    return "其他"

# ================== 文件处理 ==================
def process_excel(input_path: str, output_path: str):
    """改进的文件处理流程"""
    try:
        # 读取文件
        df = pd.read_excel(input_path, dtype=str)
        print("成功读取文件，前5行数据预览:")
        print(df.head())
        
        # 自动选择术语列
        term_col = df.columns[0]  # 直接选择第一列
        print(f"正在处理列: {term_col}")
        
        # 带进度显示的分类处理
        total = len(df)
        results = []
        
        with tqdm(total=total, desc="分类进度") as pbar:
            for idx, row in df.iterrows():
                term = row[term_col]
                category = auto_classify_term(term)
                results.append(category)
                pbar.update(1)
        
        df["自动分类"] = results
        
        # 保存结果
        df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"\n处理完成！结果已保存到: {output_path}")
    
    except Exception as e:
        print(f"\n处理失败: {str(e)}")
        print("常见解决方法:")
        print("1. 安装最新依赖: pip install pandas openpyxl tqdm --upgrade")
        print("2. 确保Excel文件不是只读模式")
        print("3. 检查文件编码是否为UTF-8")

# ================== 主程序 ==================
if __name__ == "__main__":
    # 配置路径（硬编码示例）
    input_file = Path("/workspace/DataProcess/carbon_terms.xlsx")
    output_file = Path("/workspace/DataProcess/classified_terms.xlsx")
    
    # 输入验证
    if not input_file.exists():
        print(f"错误：输入文件不存在 {input_file.resolve()}")
        exit(1)
        
    # 执行处理
    process_excel(str(input_file), str(output_file))