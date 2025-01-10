from langchain_huggingface import HuggingFaceEmbeddings
import torch
from typing import List
from pydantic import Field

class CustomHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    Pydantic是一个用于数据验证和设置管理的Python库，主要用于：
    1. 数据验证：确保输入数据符合预期类型和约束
    2. 设置管理：管理配置项和默认值
    3. 文档生成：自动生成API文档
    4. 类型提示：增强代码可读性和IDE支持

    使用场景：
    1. 配置管理：管理模型参数
    2. API输入验证：验证外部输入
    3. 数据模型：定义复杂数据结构

    可以在实例化时传入target_dim，例如：
    embeddings = CustomHuggingFaceEmbeddings(target_dim=2048)
    这将覆盖默认的1536值
    """
    target_dim: int = Field(default=1536)  # 使用pydantic的Field声明字段
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档转换为嵌入向量"""
        embeddings = super().embed_documents(texts)
        adjusted_embeddings = []
        for emb in embeddings:
            emb_tensor = torch.tensor(emb)
            if len(emb) < self.target_dim:
                padding = torch.zeros(self.target_dim - len(emb))
                adjusted_emb = torch.cat([emb_tensor, padding])
            else:
                adjusted_emb = emb_tensor[:self.target_dim]
            adjusted_embeddings.append(adjusted_emb.tolist())
        return adjusted_embeddings

    def embed_query(self, text: str) -> List[float]:
        """将单个查询文本转换为嵌入向量"""
        embedding = super().embed_query(text)
        emb_tensor = torch.tensor(embedding)
        if len(embedding) < self.target_dim:
            padding = torch.zeros(self.target_dim - len(embedding))
            adjusted_emb = torch.cat([emb_tensor, padding])
        else:
            adjusted_emb = emb_tensor[:self.target_dim]
        return adjusted_emb.tolist()


# 初始化自定义嵌入模型
embeddings = CustomHuggingFaceEmbeddings(target_dim=2048)

# 使用示例
texts = ["这是第一个文本", "这是第二个文本"]
doc_embeddings = embeddings.embed_documents(texts)
query_embedding = embeddings.embed_query("这是查询文本")

# 验证维度
print(len(doc_embeddings[0]))  # 应该输出 1536
print(len(query_embedding))     # 应该输出 1536