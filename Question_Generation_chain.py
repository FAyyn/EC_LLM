import os
import re
import json
import time
import hashlib
import random
import textwrap
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import ollama
from threading import Lock
import argparse
from concurrent.futures import ThreadPoolExecutor

# 系统配置（保持不变）
CARBON_CONFIG = {
    "model": "Qwen2.5:14b",
    "output_file": "carbon_questions_new_1.json",
    "debug_log": "carbon_debug.log",
    "encoding": "utf-8",
    "min_questions": 5,
    "max_questions": 15,
    "max_retries": 5,
    "temperature": 0.7,
    "max_context_length": 150000,
    "term_excel": "/workspace/DataProcess/carbon_terms.xlsx",
    "required_terms": 2,
    "min_file_size": 1024,
    "codec_list": ["utf-8", "gb18030", "latin1"],
    "request_interval": 1.5,
    "allowed_ext": [".md", ".txt", ".docx"],
    "enable_debug": False
}

class TermSystem:
    """术语管理系统（保持不变）"""
    def __init__(self):
        self.zh_terms = []
        self.en_map = {}
        self._load_terms()
    
    def _load_terms(self):
        try:
            df = pd.read_excel(
                CARBON_CONFIG["term_excel"],
                engine="openpyxl",
                header=None,
                usecols=[0, 1],
                dtype=str
            )
            self.zh_terms = [str(row[0]).strip() for _, row in df.iterrows() if str(row[0]).strip()]
            self.en_map = {str(row[0]).strip(): str(row[1]).strip() for _, row in df.iterrows() 
                         if str(row[0]).strip() and pd.notna(row[1])}
            if not self.zh_terms:
                raise ValueError("术语表为空，请检查Excel文件内容")
            if CARBON_CONFIG["enable_debug"]:
                print(f"[DEBUG] 加载术语示例：{self.zh_terms[:5]}...")
        except Exception as e:
            error_msg = f"术语表加载失败：{str(e)}"
            raise RuntimeError(error_msg)

class QuestionEngine:
    """问题生成引擎"""
    CLASSIFICATION_SYSTEM = {
        # ESG体系
        "ESG相关论文": {"domain":"双碳","category":"ESG体系","sub_category": "ESG相关论文"},
        "标准和评级": {"domain":"双碳","category":"ESG体系","sub_category": "标准和评级"},
        "企业白皮书和指南": {"domain":"双碳","category":"ESG体系","sub_category": "企业白皮书和指南"},
        "上市企业报告": {"domain":"双碳","category":"ESG体系","sub_category": "上市企业报告"},
        "案例和研究报告": {"domain":"双碳","category":"ESG体系","sub_category": "案例和研究报告"},

        # CCER
        "标准和规定": {"domain":"双碳","category":"CCER","sub_category": "标准和规定"},
        "计量和检测": {"domain":"双碳","category":"CCER","sub_category": "计量和检测"},
        "项目案例": {"domain":"双碳","category":"CCER","sub_category": "项目案例"},
        "战略和制度": {"domain":"双碳","category":"CCER","sub_category": "战略和制度"},

        # 碳普惠
        "碳普惠方法学": {"domain":"双碳","category":"碳普惠","sub_category": "碳普惠方法学"},
        "碳普惠研究报告": {"domain":"双碳","category":"碳普惠","sub_category": "碳普惠研究报告"},
        "碳普惠政策": {"domain":"双碳","category":"碳普惠","sub_category": "碳普惠政策"},

        # 双碳政策
        "地方政策": {"domain":"双碳","category":"双碳政策","sub_category": "地方政策"},
        "国家政策": {"domain":"双碳","category":"双碳政策","sub_category": "国家政策"},

        # 双碳工具
        "工具模板与手册": {"domain":"双碳","category":"双碳工具","sub_category": "工具模板与手册"},
        "交易与金融": {"domain":"双碳","category":"双碳工具","sub_category": "交易与金融"},
        "市场数据与分析": {"domain":"双碳","category":"双碳工具","sub_category": "市场数据与分析"},

        # 双碳方案
        "近零碳区域": {"domain":"双碳","category":"双碳方案","sub_category": "近零碳和零碳区域"},
        "能源协同": {"domain":"双碳","category":"双碳方案","sub_category": "能源互动与协同"},
        "双碳标准": {"domain":"双碳","category":"双碳方案","sub_category": "双碳标准与指南"},
        "研究报告": {"domain":"双碳","category":"双碳方案","sub_category": "双碳研究报告"},
        "生态保护": {"domain":"双碳","category":"双碳方案","sub_category": "碳汇与生态保护"},
        "碳中和案例": {"domain":"双碳","category":"双碳方案","sub_category": "碳中和项目方案案例"},
        "能源优化": {"domain":"双碳","category":"双碳方案","sub_category": "综合能源优化"},

        # 绿色发展
        "可再生能源": {"domain":"双碳","category":"绿色发展","sub_category": "可再生能源发展"},
        "绿色建筑": {"domain":"双碳","category":"绿色发展","sub_category": "绿色建筑"},
        "绿色金融": {"domain":"双碳","category":"绿色发展","sub_category": "绿色金融"},
        "绿色转型": {"domain":"双碳","category":"绿色发展","sub_category": "绿色转型"},

        # 其他
        "行业研报": {"domain":"双碳","category":"行业研报","sub_category": "行业研报"},
        "氢能储能": {"domain":"双碳","category":"氢能储能","sub_category": "氢能储能"},
    }

    CLASSIFICATION_KEYWORDS = {
        # ESG体系
        "ESG相关论文": ["研究", "论文", "文献", "实证", "学术", "期刊", "综述", "模型", "方法论", "数据分析"],
        "标准和评级": ["标准", "评级", "认证", "ISO", "评定", "SA8000", "GRI", "评估体系", "等级划分", "信用评级"],
        "企业白皮书和指南": ["白皮书", "指南", "指引", "手册", "指导文件", "最佳实践", "操作规范", "行业基准", "技术文档", "框架文件"],
        "上市企业报告": ["年报", "ESG报告", "社会责任报告", "披露报告", "TCFD报告", "可持续发展报告", "整合报告", "非财务信息披露", "透明度报告"],
        "案例和研究报告": ["案例", "实践", "应用", "示范", "调研报告", "案例库", "试点项目", "成效评估", "行业白皮书", "基准分析"],

        # CCER
        "标准和规定": ["标准", "规范", "规定", "规程", "MRV体系", "备案规则", "审定流程", "合规要求", "监管框架"],
        "计量和检测": ["测量", "监测", "校准", "验证", "检测", "碳核算", "数据溯源", "仪器标定", "不确定性分析", "质量控制"],
        "项目案例": ["项目", "实施案例", "应用实例", "示范工程", "林业碳汇", "光伏减排", "甲烷回收", "能效提升项目", "CCER备案项目"],
        "战略和制度": ["战略", "规划", "制度", "管理体系", "碳中和路线图", "减排目标", "碳资产管理", "内部碳定价", "制度创新"],

        # 碳普惠
        "碳普惠方法学": ["方法学", "计算模型", "核算方法", "减排因子", "场景参数", "基准线设定", "额外性论证", "普惠场景", "算法优化"],
        "碳普惠研究报告": ["研究", "分析报告", "评估报告", "公众参与度", "平台运营", "碳积分体系", "激励机制设计", "区域试点评估"],
        "碳普惠政策": ["政策", "激励措施", "补贴", "奖励机制", "碳账户", "个人减排", "社区参与", "商业联盟", "政府合作"],

        # 双碳政策
        "地方政策": ["地方条例", "省级规划", "城市达峰方案", "区域试点", "财政补贴政策", "用能权交易", "碳排放限额"],
        "国家政策": ["国家战略", "1+N政策", "行业达峰计划", "全国碳市场", "绿色产业目录", "气候投融资", "国际合作机制"],

        # 双碳工具
        "工具模板与手册": ["碳盘查工具", "LCA软件", "计算器模板", "填报指南", "MRV手册", "核查清单", "数字化平台"],
        "交易与金融": ["碳期货", "配额拍卖", "绿色信贷", "CCER质押", "碳保险", "跨境交易", "金融机构准入"],
        "市场数据与分析": ["价格指数", "成交量分析", "履约周期", "行业配额分配", "供需预测", "政策影响评估"],

        # 双碳方案
        "近零碳区域": ["园区碳中和", "零碳建筑群", "分布式能源", "微电网设计", "碳普惠社区", "智慧能源管理"],
        "能源协同": ["源网荷储", "虚拟电厂", "多能互补", "需求响应", "储能调度", "跨区域输电"],
        "双碳标准": ["产品碳足迹", "绿色工厂评价", "碳中和认证", "碳捕集规范", "氢能安全标准"],
        "研究报告": ["技术路线图", "行业深度研究", "政策解读报告", "国际对标分析", "技术经济性评估"],
        "生态保护": ["红树林修复", "生物多样性核算", "生态补偿机制", "自然资本评估", "国土空间规划"],
        "碳中和案例": ["数据中心降碳", "零碳赛事", "航空生物燃料", "钢铁氢能冶炼", "碳移除项目"],
        "能源优化": ["余热回收", "工艺改造", "智能控制算法", "能源审计", "梯级利用"],

        # 绿色发展
        "可再生能源": ["风光储一体化", "海上风电", "绿氢制备", "生物质耦合发电", "可再生能源配额"],
        "绿色建筑": ["超低能耗", "BIM应用", "装配式建筑", "健康建筑认证", "建筑光伏一体化"],
        "绿色金融": ["转型金融", "碳金融衍生品", "ESG理财产品", "环境信息披露", "赤道原则"],
        "绿色转型": ["高耗能行业", "供应链脱碳", "循环经济模式", "清洁生产审核", "工业互联网+"],

        # 其他
        "行业研报": ["电解槽技术", "新型储能", "碳捕集成本", "绿电交易", "欧盟CBAM影响"],
        "氢能储能": ["氨氢融合", "液态储运", "燃料电池车", "固态储氢", "绿氨合成"],
    }

    def __init__(self, term_system: TermSystem):
        self.term_system = term_system
        self.lock = Lock()
        self.questions = []
        self.fingerprints = set()
        self.stats = {
            "total_files": 0,
            "success_files": 0,
            "failed_files": 0,
            "total_questions": 0,
            "valid_questions": 0,
            "duplicates": 0,
            "classification_stats": {}
        }
        self._check_model_health()

    def _check_model_health(self):
        """模型服务健康检查（保持不变）"""
        try:
            ollama.list()
            if CARBON_CONFIG["enable_debug"]:
                print("[DEBUG] 模型服务连接正常")
        except Exception as e:
            error_msg = f"模型服务不可用：{str(e)}"
            raise RuntimeError(error_msg)

    def process(self, input_dir: Path):
        """处理目录（调整为调用新的生成方法）"""
        files = self._find_files(input_dir)
        if not files:
            raise ValueError("未找到有效文件")
        if CARBON_CONFIG["enable_debug"]:
            print(f"[DEBUG] 找到{len(files)}个待处理文件")
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(tqdm(executor.map(self._process_file, files), total=len(files), desc="处理进度"))

    def _find_files(self, directory: Path) -> List[Path]:
        """查找有效文件（保持不变）"""
        valid_files = []
        for ext in CARBON_CONFIG["allowed_ext"]:
            for f in directory.rglob(f"*{ext}"):
                if f.stat().st_size >= CARBON_CONFIG["min_file_size"]:
                    valid_files.append(f)
                else:
                    self._log_skip(f, "文件过小")
        return valid_files

    def _process_file(self, file_path: Path):
        """处理单个文件（调整为调用新的生成方法）"""
        try:
            content = self._read_file(file_path)
            if not content:
                self._log_skip(file_path, "内容为空")
                return
            clean_name = self._clean_filename(file_path.stem)
            terms = self._extract_terms(content)
            
            # 使用思维链生成问题
            model_questions = self._generate_questions_with_cot(content, clean_name, terms)
            template_questions = self._generate_template_questions(terms, clean_name)
            all_questions = model_questions + template_questions
            
            # 过滤和去重
            valid_questions = self._quality_filter(all_questions)
            final_questions = self._deduplicate(valid_questions)
            
            with self.lock:
                self.questions.extend(final_questions)
                self._update_stats(len(final_questions), clean_name)

        except Exception as e:
            with self.lock:
                self.stats["failed_files"] += 1
            self._log_error(file_path, e)

    def _update_stats(self, new_questions: int, source: str):
        """更新统计信息（保持不变）"""
        self.stats["success_files"] += 1
        self.stats["total_files"] += 1
        self.stats["valid_questions"] += new_questions
        if CARBON_CONFIG["enable_debug"] and new_questions > 0:
            print(f"[DEBUG] 文件 {source} 生成{new_questions}个有效问题")

    def _read_file(self, file_path: Path) -> Optional[str]:
        """安全读取文件内容（保持不变）"""
        for codec in CARBON_CONFIG["codec_list"]:
            try:
                with open(file_path, "r", encoding=codec) as f:
                    content = f.read(CARBON_CONFIG["max_context_length"])
                return self._clean_content(content)
            except Exception as e:
                continue
        return None

    def _clean_content(self, content: str) -> str:
        """内容清洗处理（保持不变）"""
        content = re.sub(r"[\x00-\x1F\uFEFF]", " ", content)
        return re.sub(r"\s+", " ", content).strip()

    def _extract_terms(self, content: str) -> List[str]:
        """多维度术语提取（保持不变）"""
        core_terms = [term for term in self.term_system.zh_terms if term in content]
        tech_terms = [m.group() for m in re.finditer(
            r"[\u4e00-\u9fff]+\d+[%°\u4e00-\u9fff]*|[\u4e00-\u9fff]+率|[\u4e00-\u9fff]+量", content)]
        if not core_terms + tech_terms:
            nouns = re.findall(r"[\u4e00-\u9fff]{2,5}?(?:体系|参数|标准)", content)
            return list(set(nouns))[:3]
        return list(set(core_terms + tech_terms))

    ### 新增方法：使用思维链生成问题
    def _generate_questions_with_cot(self, content: str, source: str, terms: List[str]) -> List[Dict]:
        """使用思维链生成问题"""
        prompt = self._build_cot_prompt(content, source, terms)
        try:
            response = ollama.generate(
                model=CARBON_CONFIG["model"],
                prompt=prompt,
                options={
                    "temperature": CARBON_CONFIG["temperature"],
                    "num_predict": 8196  # 增加预测长度以适应思维链输出
                }
            )
            return self._parse_cot_response(response["response"], source)
        except Exception as e:
            if CARBON_CONFIG["enable_debug"]:
                print(f"[ERROR] 模型生成失败：{str(e)}")
            return []

    ### 新增方法：构建思维链提示词
    def _build_cot_prompt(self, content: str, source: str, terms: List[str]) -> str:
        """构建思维链提示词"""
        term_instruction = "请使用以下术语：" + "、".join(terms[:5]) if terms else "请关注技术参数"
        return textwrap.dedent(f"""
        【任务】生成专业技术问题
        【要求】
        1. {term_instruction}
        2. 包含具体数值或技术指标
        3. 使用疑问句式
        
        【内容片段】
        {textwrap.shorten(content, width=4000, placeholder='...')}
        
        【思维链】
        1. **提取上下文信息**：从内容片段中提取关键的上下文信息。
        2. **总结文章内容**：简要总结文章的大致内容。
        3. **判断关键词和来源的地位**：分析关键词“{', '.join(terms[:5])}”和来源“{source}”在文章中的地位和关联性。
        4. **生成问题**：基于上述信息，生成3-5个专业技术问题，问题的数量应根据文章内容的丰富程度而定。
        
        请按照思维链的步骤进行思考，并在最后输出生成的问题，格式如下：
        - 问题1：具体问题内容
        - 问题2：具体问题内容
        - 问题3：具体问题内容
        """)

    ### 新增方法：解析思维链响应
    def _parse_cot_response(self, raw: str, source: str) -> List[Dict]:
        """解析思维链响应"""
        questions = []
        raw_text = str(raw).strip()
        # 提取以“问题X：”开头的行
        for line in re.split(r"\n+", raw_text):
            if re.match(r"^-?\s*问题\d*：", line):
                question_text = re.sub(r"^-?\s*问题\d*：\s*", "", line).strip()
                if self._is_valid_question(question_text):
                    questions.append(self._create_question(question_text, source))
        return questions

    def _generate_template_questions(self, terms: List[str], source: str) -> List[Dict]:
        """模板生成系统（修改后）"""
        templates = [
            ("term", "如何验证{term}的准确性？"),
            ("term", "使用{term}时需要满足哪些基本标准？"),
            ("term", "实施{term}监测时常见的误差来源有哪些？"),
            ("term", "如何减少{term}在实际应用中的偏差？"),
            ("term", "如何通过{term}评估碳排放？"),
            ("term", "{term}的数据如何支持决策或报告？"),
            ("term", "{term}与其他碳排放指标之间如何转换？"),
            ("term", "{term}如何与其他标准或方法配合使用？"),
            ("term", "{term}需要满足哪些政策或法规要求？"),
            ("term", "如何通过{term}满足碳排放监管需求？"),
        ]
        questions = []
        for _ in range(3):
            try:
                template_type, template = random.choice(templates)
                if template_type == "term":
                    if not terms:
                        if CARBON_CONFIG["enable_debug"]:
                            print(f"[WARNING] 文件 {source} 中未提取到术语，跳过需要 term 的模板")
                        continue  # 跳过需要 term 的模板
                    term = random.choice(terms)
                    question = template.format(source=source, term=term)
                else:
                    question = template.format(source=source)
                if "{" in question or "}" in question:
                    continue
                questions.append(self._create_question(question, source))
            except Exception as e:
                self._log_error(Path(source), e)
        return questions
    def _create_question(self, text: str, source: str) -> Dict:
        """创建问题对象（保持不变）"""
        content = re.sub(r"\s+", " ", text).strip()
        classification = self._classify_question(content)
        class_path = f"{classification['domain']}-{classification['category']}-{classification['sub_category']}"
        self.stats["classification_stats"][class_path] = self.stats["classification_stats"].get(class_path, 0) + 1
        return {
            "content": content,
            "source": source,
            "classification": classification,
            "fingerprint": hashlib.md5(content.encode()).hexdigest()
        }

    def _classify_question(self, question: str) -> Dict:
        """三级分类逻辑（保持不变）"""
        question_lower = question.lower()
        for sub_cat, keywords in self.CLASSIFICATION_KEYWORDS.items():
            if any(kw in question_lower for kw in keywords):
                class_info = self.CLASSIFICATION_SYSTEM.get(sub_cat, {})
                return {
                    "domain": class_info.get("domain", "其他"),
                    "category": class_info.get("category", "其他"),
                    "sub_category": sub_cat
                }
        tech_keywords = {
            "技术标准": ["要求", "规范", "参数", "标准", "准则", "规程", "指南"],
            "实施方法": ["如何", "步骤", "流程", "计算", "评估", "测量", "监测"],
            "监测验证": ["验证", "核查", "校准", "审计", "认证", "审定"],
            "数据管理": ["数据采集", "数据处理", "数据分析", "数据报告"],
            "政策法规": ["政策", "法规", "标准", "指南"],
            "技术创新": ["技术研发", "技术应用", "技术推广"],
            "经济分析": ["成本效益分析", "投资回报率", "经济模型"],
            "社会影响": ["社会责任", "公众参与", "社会效益"],
            "国际合作": ["国际标准", "国际认证", "国际项目"],
            "教育培训": ["培训课程", "教育项目", "技能认证"]
        }
        for cat, kws in tech_keywords.items():
            if any(kw in question_lower for kw in kws):
                return {"domain": "双碳", "category": "通用技术", "sub_category": cat}
        return {"domain": "双碳", "category": "其他", "sub_category": "未分类"}

    def _is_valid_question(self, question: str) -> bool:
        """问题有效性验证（保持不变）"""
        return (
            15 <= len(question) <= 200 and
            any(question.endswith(c) for c in ("？", "?", "：", ":")) and
            not re.search(r"[<>\[\]]", question)
        )

    def _quality_filter(self, questions: List[Dict]) -> List[Dict]:
        """质量过滤（保持不变）"""
        return [q for q in questions if self._is_valid_question(q["content"])]

    def _deduplicate(self, questions: List[Dict]) -> List[Dict]:
        """深度去重（保持不变）"""
        unique = []
        for q in questions:
            fp = q["fingerprint"]
            if fp not in self.fingerprints:
                self.fingerprints.add(fp)
                unique.append({k: v for k, v in q.items() if k != "fingerprint"})
            else:
                self.stats["duplicates"] += 1
        return unique

    def _clean_filename(self, filename: str) -> str:
        """文件名清洗（保持不变）"""
        return re.sub(r'[\\/*?:"<>|]', "", filename).strip()

    def _log_skip(self, file_path: Path, reason: str):
        """记录跳过原因（保持不变）"""
        log_msg = f"[SKIP] {file_path.name} - {reason}\n"
        with open(CARBON_CONFIG["debug_log"], "a") as f:
            f.write(log_msg)

    def _log_error(self, file_path: Path, error: Exception):
        """记录错误日志（保持不变）"""
        log_msg = (
            f"[ERROR] {time.ctime()} {file_path.name}\n"
            f"类型: {type(error).__name__}\n"
            f"详情: {str(error)[:300]}\n"
            "------------------------\n"
        )
        with open(CARBON_CONFIG["debug_log"], "a") as f:
            f.write(log_msg)

    def generate_report(self) -> Dict:
        """生成统计报告（保持不变）"""
        return {
            "metadata": {
                "total_files": self.stats["total_files"],
                "success_rate": f"{self.stats['success_files']/self.stats['total_files']:.1%}" if self.stats['total_files'] > 0 else "0%",
                "valid_questions": self.stats["valid_questions"],
                "duplicates": self.stats["duplicates"],
                "classification_distribution": self.stats["classification_stats"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "questions": self.questions
        }

def main():
    """主函数（保持不变）"""
    parser = argparse.ArgumentParser(description="能碳问题生成器 v7.1.2")
    parser.add_argument("-i", "--input", required=True, help="输入目录路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    CARBON_CONFIG["enable_debug"] = args.debug

    try:
        start_time = time.time()
        input_path = Path(args.input)
        term_system = TermSystem()
        engine = QuestionEngine(term_system)
        engine.process(input_path)
        with open(CARBON_CONFIG["output_file"], "w", encoding="utf-8") as f:
            json.dump(engine.generate_report(), f, ensure_ascii=False, indent=2)
        print("\n处理摘要：")
        print(f"• 处理文件数：{engine.stats['total_files']}")
        print(f"• 成功文件数：{engine.stats['success_files']}")
        print(f"• 有效问题数：{engine.stats['valid_questions']}")
        print(f"• 分类分布：")
        for cls, count in engine.stats["classification_stats"].items():
            print(f"  - {cls}: {count}")
        print(f"• 总耗时：{time.time()-start_time:.1f}秒")
    except Exception as e:
        print(f"\n处理异常：{str(e)}")
        if CARBON_CONFIG["enable_debug"]:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if not Path(CARBON_CONFIG["term_excel"]).exists():
        print(f"关键错误：术语表文件不存在 {CARBON_CONFIG['term_excel']}")
        exit(1)
    main()