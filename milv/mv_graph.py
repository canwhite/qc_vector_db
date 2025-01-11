import sys
import os
import pathlib import Path
root_path = Path(__file__).parent.parent
sys.path.append(root_path)
from llm import get_completion_from_prompt,get_completion_from_messages
from HuggingFaceEmbeddings import HuggingFaceEmbeddings

import numpy as np
from collections import defaultdict
# scipy 是一个用于科学计算和技术计算的 Python 库，主要功能包括：
# 1. 数值计算：提供高效的数值积分、优化、插值等算法
# 2. 线性代数：包含稀疏矩阵运算、特征值计算等
# 3. 信号处理：提供傅里叶变换、滤波器设计等工具
# 4. 统计函数：包含概率分布、假设检验等统计方法
# 5. 图像处理：提供图像滤波、形态学操作等功能
# 6. 稀疏矩阵：支持高效存储和操作大型稀疏矩阵
# 7. 特殊函数：包含贝塞尔函数、伽马函数等数学特殊函数
# 在本项目中主要用于处理图数据中的稀疏矩阵运算
from scipy.sparse import csr_matrix
from pymilvus import MilvusClient
from tqdm import tqdm
"""
核心数据结构（如 Document、BaseMessage）。
语言模型接口（如 BaseLanguageModel、LLM）。
提示模板（如 PromptTemplate）。
链式操作（如 Chain、LLMChain）。
代理和工具（如 Agent、Tool）。
记忆模块（如 ConversationBufferMemory）。
嵌入模型（如 Embeddings）。
缓存机制（如 BaseCache）。
"""
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

nano_dataset = [
    {
        "passage": "Jakob Bernoulli (1654–1705): Jakob was one of the earliest members of the Bernoulli family to gain prominence in mathematics. He made significant contributions to calculus, particularly in the development of the theory of probability. He is known for the Bernoulli numbers and the Bernoulli theorem, a precursor to the law of large numbers. He was the older brother of Johann Bernoulli, another influential mathematician, and the two had a complex relationship that involved both collaboration and rivalry.",
        "triplets": [
            ["Jakob Bernoulli", "made significant contributions to", "calculus"],
            [
                "Jakob Bernoulli",
                "made significant contributions to",
                "the theory of probability",
            ],
            ["Jakob Bernoulli", "is known for", "the Bernoulli numbers"],
            ["Jakob Bernoulli", "is known for", "the Bernoulli theorem"],
            ["The Bernoulli theorem", "is a precursor to", "the law of large numbers"],
            ["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"],
        ],
    },
    {
        "passage": "Johann Bernoulli (1667–1748): Johann, Jakob’s younger brother, was also a major figure in the development of calculus. He worked on infinitesimal calculus and was instrumental in spreading the ideas of Leibniz across Europe. Johann also contributed to the calculus of variations and was known for his work on the brachistochrone problem, which is the curve of fastest descent between two points.",
        "triplets": [
            [
                "Johann Bernoulli",
                "was a major figure of",
                "the development of calculus",
            ],
            ["Johann Bernoulli", "was", "Jakob's younger brother"],
            ["Johann Bernoulli", "worked on", "infinitesimal calculus"],
            ["Johann Bernoulli", "was instrumental in spreading", "Leibniz's ideas"],
            ["Johann Bernoulli", "contributed to", "the calculus of variations"],
            ["Johann Bernoulli", "was known for", "the brachistochrone problem"],
        ],
    },
    {
        "passage": "Daniel Bernoulli (1700–1782): The son of Johann Bernoulli, Daniel made major contributions to fluid dynamics, probability, and statistics. He is most famous for Bernoulli’s principle, which describes the behavior of fluid flow and is fundamental to the understanding of aerodynamics.",
        "triplets": [
            ["Daniel Bernoulli", "was the son of", "Johann Bernoulli"],
            ["Daniel Bernoulli", "made major contributions to", "fluid dynamics"],
            ["Daniel Bernoulli", "made major contributions to", "probability"],
            ["Daniel Bernoulli", "made major contributions to", "statistics"],
            ["Daniel Bernoulli", "is most famous for", "Bernoulli’s principle"],
            [
                "Bernoulli’s principle",
                "is fundamental to",
                "the understanding of aerodynamics",
            ],
        ],
    },
    {
        "passage": "Leonhard Euler (1707–1783) was one of the greatest mathematicians of all time, and his relationship with the Bernoulli family was significant. Euler was born in Basel and was a student of Johann Bernoulli, who recognized his exceptional talent and mentored him in mathematics. Johann Bernoulli’s influence on Euler was profound, and Euler later expanded upon many of the ideas and methods he learned from the Bernoullis.",
        "triplets": [
            [
                "Leonhard Euler",
                "had a significant relationship with",
                "the Bernoulli family",
            ],
            ["leonhard Euler", "was born in", "Basel"],
            ["Leonhard Euler", "was a student of", "Johann Bernoulli"],
            ["Johann Bernoulli's influence", "was profound on", "Euler"],
        ],
    },
]

# 这两个defaultdict用于建立实体、关系和段落之间的映射关系
# defaultdict是collections模块中的一个特殊字典，当访问不存在的键时会自动创建默认值
# 这里使用list作为默认值，方便后续添加多个值

# entityid_2_relationids: 用于存储实体ID到关系ID的映射
# 例如：entityid_2_relationids[实体A] = [关系1, 关系2,...]
# 表示实体A参与了关系1和关系2

# relationid_2_passageids: 用于存储关系ID到段落ID的映射  
# 例如：relationid_2_passageids[关系1] = [段落1, 段落2,...]
# 表示关系1出现在段落1和段落2中

# 这种数据结构设计可以：
# 1. 快速查找某个实体参与的所有关系
# 2. 快速查找某个关系出现在哪些段落
# 3. 支持一对多的映射关系
# 4. 避免手动初始化空列表的麻烦

entityid_2_relationids = defaultdict(list)
relationid_2_passageids = defaultdict(list)


#建立关系
entities = []
relations = []
passages = []

#处理这些关系












