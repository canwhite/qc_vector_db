#image similarity 
import numpy as np
print(np.__version__) 

import torch
import timm
from sklearn.preprocessing import normalize # 
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# in order to add llm package
import sys as system
from pathlib import Path 
root_path = Path(__file__).parent.parent
root_path = str(root_path)
system.path.append(root_path)


import torch
from PIL import Image # 图片显示
import timm # timm提供图片的剪切等操作
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform 
from sklearn.preprocessing import normalize #预测结果向量的归一化

from pymilvus import MilvusClient
import os 


# display
from IPython.display import display

# fulfil a feature extractor
class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()
        # input_size 是模型处理图像时所需的输入尺寸
        # 它的重要性体现在：
        # 1. 确保输入图像尺寸与模型训练时一致，保证特征提取的准确性
        self.input_size = self.model.default_cfg["input_size"]

        # Get the preprocessing function provided by TIMM for the model
        # 做些预处理
        config = resolve_data_config({}, model=modelname)
        self.preprocess = create_transform(**config)

    # __call__ 方法会在以下情况被调用：
    # 1. 当实例对象被直接调用时，例如：extractor = FeatureExtractor('resnet50'); extractor('image.jpg')
    def __call__(self, imagepath):

        input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            # 得到预测结果
            output = self.model(input_tensor)
        #Extract the feature vector
        feature_vector = output.squeeze().numpy()

        # 这段代码的主要功能是对提取的特征向量进行归一化处理
        # - 消除特征向量长度对相似度计算的影响
        # - 使不同特征向量之间的比较更加公平
        # - 提高后续向量检索的准确性

        # L2范式归一化后的输出结果范围是[0, 1]
        # 具体来说：
        # 1. 每个特征向量都会被归一化为单位向量（长度为1）
        # 2. 向量中的每个元素值都会落在0到1之间
        # 3. 所有元素的平方和为1
        # 4. 这种归一化方式特别适合用于余弦相似度计算
        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()


image_db_path = str(Path(__file__).parent.parent / "data/image_db.db")

# Set up a Milvus client
client = MilvusClient(image_db_path)


# Create a collection in quick setup mode
if client.has_collection(collection_name="image_embeddings"):
    client.drop_collection(collection_name="image_embeddings")

# 这里默认使用512维度的原因：
# 1. 512是一个常用的特征向量维度大小，在图像处理领域被广泛使用
# 2. 大多数预训练模型（如ResNet、EfficientNet等）的输出特征维度都在512左右
# 3. 512维在计算效率和特征表达能力之间提供了良好的平衡
# 4. 与Milvus的默认配置和性能优化相匹配
# 5. 如果需要调整维度，可以根据实际使用的特征提取模型进行修改
print(f"使用512维作为默认向量维度，因为：")
print("1. 这是图像特征提取的常用维度大小")
print("2. 与大多数预训练模型的输出维度匹配")
print("3. 在计算效率和特征表达能力之间提供良好平衡")

client.create_collection(
    collection_name="image_embeddings",
    vector_field_name="vector",
    dimension=512,
    auto_id=True,
    enable_dynamic_field=True,
    metric_type="COSINE", # 余弦
)

# insert the Embeddings to Milvus
extractor = FeatureExtractor("resnet34")


train_path = str(Path(__file__).parent.parent / "train")


insert = True
if insert is True:
    # os.walk() 是一个用于遍历目录树的函数，返回一个三元组 (dirpath, dirnames, filenames)
    # 其中：
    # 1. dirpath: 当前遍历的目录路径（字符串）
    # 2. dirnames: 当前目录下的子目录列表（列表）
    # 3. filenames: 当前目录下的文件列表（列表）
    # 它会递归遍历指定目录及其所有子目录
    # 在本例中，我们使用它来遍历训练数据目录中的所有图像文件
    # 对于每个找到的.JPEG文件，我们提取其特征向量并插入到Milvus中
    for dirpath, foldername, filenames in os.walk(train_path):
        for filename in filenames:
            if filename.endswith(".JPEG"):
                filepath = dirpath + "/" + filename
                image_embedding = extractor(filepath)
                #insert
                client.insert(
                    "image_embeddings",
                    {"vector": image_embedding, "filename": filepath},
                )


# 最终看一下我们的结果

query_image = train_path +  "/Afghan_hound/n02088094_1045.JPEG"

results = client.search(
    "image_embeddings",
    data=[extractor(query_image)],
    output_fields=["filename"],
    search_params={"metric_type": "COSINE"},
)

#先做一层预处理
images = []
for result in results:
    for hit in result[:10]:
        filename = hit["entity"]["filename"]
        img = Image.open(filename)
        img = img.resize((150, 150))
        images.append(img)

# 再展示
width = 150 * 5
height = 150 * 2
concatenated_image = Image.new("RGB", (width, height))

for idx, img in enumerate(images):
    # 将图像粘贴到拼接图像上
    # 计算每个图像在拼接图像中的位置：
    # - x坐标：idx % 5 * 150 （每行5个图像，每个图像宽150像素）
    # - y坐标：idx // 5 * 150 （每列2个图像，每个图像高150像素）
    
    # x = idx % 5 计算当前图像在行中的位置（0-4）
    # y = idx // 5 计算当前图像在列中的位置（0-1）
    # 使用PIL的paste方法将图像粘贴到指定位置

    x = idx % 5
    y = idx // 5
    concatenated_image.paste(img, (x * 150, y * 150))


# 使用PIL的show方法展示图片
print("Query Image:")
Image.open(query_image).resize((150, 150)).show()

print("\nSearch Results:")
concatenated_image.show()

# diplay需要在框架上显示
# display("query")
# display(Image.open(query_image).resize((150, 150)))
# display("results")
# display(concatenated_image)