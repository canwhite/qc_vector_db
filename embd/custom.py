from sentence_transformers import SentenceTransformer
import os
import sys
from pathlib import Path
#expand dim
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize


root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

model_path = root_path + "/embd_model"

# 检查模型路径是否存在且为空
if not os.path.exists(model_path) or not os.listdir(model_path):
    # 如果路径不存在或为空，则创建目录
    os.makedirs(model_path, exist_ok=True)

    # 指定要下载的模型名称
    model_name = "sentence-transformers/all-mpnet-base-v2"

    # 下载模型
    model = SentenceTransformer(model_name)

    # 保存模型到指定路径
    model.save(model_path)



class CustomTransformerEmbeddings:
    def __init__(self, target_dim=1536, interpolation_method='cubic', normalize_output=True):
        """
        初始化嵌入模型
        
        :param target_dim: 目标维度
        :param interpolation_method: 插值方法 ('linear', 'cubic', 'quadratic', 'akima')
        :param normalize_output: 是否对输出进行归一化
        """
        self.model = SentenceTransformer(model_path)
        self.target_dim = target_dim
        self.original_dim = None
        self.interpolation_method = interpolation_method
        self.normalize_output = normalize_output
        
    def _resize_embedding(self, embedding):

        if self.original_dim is None:
            # 获取模型的原始维度
            test_embedding = self.model.encode(["test"])[0]
            self.original_dim = len(test_embedding)
        
        if self.original_dim == self.target_dim:
            return embedding
            
        # 将输入向量转换为numpy数组并确保为浮点类型
        embedding = np.array(embedding, dtype=np.float64)
        
        # 创建更密集的坐标点以提高精度
        # np.linspace用于在指定范围内生成等间隔的数值序列
        # 参数说明：
        # start: 序列的起始值
        # stop: 序列的结束值
        # num: 要生成的样本数量
        # 这里我们生成从0到1的等间隔点，用于插值计算
        # original_points: 原始维度对应的坐标点
        # target_points: 目标维度对应的坐标点
        # 这些点将作为插值函数的输入和输出坐标
        original_points = np.linspace(0, 1, self.original_dim)
        target_points = np.linspace(0, 1, self.target_dim)
        
        try:
            # 尝试使用指定的插值方法
            if self.interpolation_method == 'akima':
                from scipy.interpolate import Akima1DInterpolator
                interpolator = Akima1DInterpolator(original_points, embedding)
            else:
                interpolator = interp1d(original_points, embedding, 
                                      kind=self.interpolation_method,
                                      bounds_error=False, 
                                      fill_value="extrapolate")
            
            resized_embedding = interpolator(target_points)
            
        except ValueError:
            # 如果指定方法失败，回退到线性插值
            interpolator = interp1d(original_points, embedding, 
                                  kind='linear',
                                  bounds_error=False, 
                                  fill_value="extrapolate")
            resized_embedding = interpolator(target_points)
        
        # 处理可能的数值不稳定性
        resized_embedding = np.nan_to_num(resized_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 可选的归一化处理，归一化处理是将数据进行线形变化
        # 线性变换的目的是将数据映射到新的空间，通常用于：
        # 1. 统一数据尺度：使不同特征具有可比性
        # 2. 提高模型性能：某些算法对数据尺度敏感
        # 3. 加速收敛：归一化数据可以加快优化过程
        
        # L2归一化（欧几里得范数）实现方式：
        # 1. 计算向量的L2范数：sqrt(sum(x_i^2))
        # 2. 将向量每个元素除以L2范数
        # 3. 结果向量的L2范数将为1，即单位向量
        
        # 数学公式：
        # x_normalized = x / ||x||_2
        # 其中 ||x||_2 = sqrt(x1^2 + x2^2 + ... + xn^2)
        
        # 这种变换保持了向量的方向信息，同时将长度归一化
        # 在嵌入空间中，这有助于比较不同长度向量的相似性
        
        if self.normalize_output:
            resized_embedding = normalize(resized_embedding.reshape(1, -1), norm='l2')[0]
        
        return resized_embedding
        
    def embed_documents(self, texts: list) -> list:
        """
        生成多个文本的嵌入向量。

        :param texts: 输入文本列表
        :return: 调整维度后的嵌入向量列表
        """
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        
        # 使用向量化操作处理批量数据
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.astype(np.float64)
        
        resized_embeddings = [self._resize_embedding(emb) for emb in embeddings]
        return [emb.tolist() for emb in resized_embeddings]
        
    def embed_query(self, text: str) -> list:
        """
        生成单个文本的嵌入向量。

        :param text: 输入文本
        :return: 调整维度后的嵌入向量
        """
        embedding = self.model.encode(text)
        resized_embedding = self._resize_embedding(embedding)
        return resized_embedding.tolist()

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        计算两个向量的余弦相似度
        
        :param vec1: 第一个向量
        :param vec2: 第二个向量
        :return: 余弦相似度
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__ == "__main__":
    # 初始化本地嵌入模型
    embeddings = CustomTransformerEmbeddings()

    # 测试文档嵌入
    test_texts = ["这是一个测试文本", "这是另一个测试文本"]
    print("测试文档嵌入...")
    doc_embeddings = embeddings.embed_documents(test_texts)
    print(f"文档嵌入维度: {len(doc_embeddings[0])}")
    print(f"文档数量: {len(doc_embeddings)}")

    # 测试查询嵌入
    print("\n测试查询嵌入...")
    query_text = "这是一个查询测试"
    query_embedding = embeddings.embed_query(query_text)
    print(f"查询嵌入维度: {len(query_embedding)}")

    # 验证结果
    assert len(doc_embeddings) == len(test_texts), "文档嵌入数量不匹配"
    assert len(doc_embeddings[0]) == len(query_embedding), "嵌入维度不一致"
    print("\n所有测试通过！")
