from langchain_huggingface import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn
#TODO，后期可以验证
# 加载预训练的模型和分词器
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 定义投影层，将768维向量映射到1536维
projection_layer = nn.Linear(768, 1536)

def emb_texts(texts):
    # 对输入文本进行分词并转换为模型输入格式
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # 获取模型的输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取最后一层的隐藏状态（即嵌入向量）
    last_hidden_state = outputs.last_hidden_state
    
    # 对每个句子的所有token的嵌入向量进行平均，得到句子的嵌入向量
    embeddings = torch.mean(last_hidden_state, dim=1)
    
    # 使用投影层将768维向量扩展到1536维
    embeddings_1536 = projection_layer(embeddings)
    
    # 将结果转换为列表形式并返回
    return embeddings_1536.tolist()


# 测试代码
if __name__ == "__main__":
    # 测试文本
    test_texts = [
        "This is a test sentence",
        "Another example for testing",
        "The quick brown fox jumps over the lazy dog"
    ]
    
    # 获取1536维的嵌入向量
    embeddings = emb_texts(test_texts)
    
    # 打印测试结果
    print("测试结果：")
    print(f"嵌入向量维度: {len(embeddings[0])}")  # 应该输出1536
    print(f"前10个维度值示例：")
    for i, emb in enumerate(embeddings):
        print(f"文本{i+1}的前10个维度值：{emb[:10]}")
    
    # 计算文本之间的相似度
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)
    print("\n文本相似度矩阵：")
    print(similarity_matrix)
    
    # 查询示例
    query_text = "A simple test query"
    query_embedding = emb_texts([query_text])[0]
    print(f"\n查询文本的嵌入维度: {len(query_embedding)}")
    
    # 计算查询与测试文本的相似度
    query_similarities = cosine_similarity([query_embedding], embeddings)
    print("\n查询与测试文本的相似度：")
    for i, sim in enumerate(query_similarities[0]):
        print(f"与文本{i+1}的相似度: {sim:.4f}")


