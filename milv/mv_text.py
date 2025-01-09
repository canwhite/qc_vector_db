from pymilvus import MilvusClient
from pathlib import Path 

from glob import glob
from langchain_huggingface import HuggingFaceEmbeddings
# tqdm display progress 
from tqdm import tqdm
# add root path, do not need specific module path 
import sys
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)  # 添加 utils 包的路径
from llm import get_completion_from_messages
import json


"""use milvus to achieve RAG"""

# get db path 
dbPath = str(Path(__file__).parent.parent / "data/milvus_demo.db")

# create milvus  client
milvus_client = MilvusClient(dbPath)

#--get text content 
text_lines = []
docs_path = str(Path(root_path) / "milvus_docs/en/faq/*.md")

for file_path in glob(docs_path, recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")


#--embed text, use hugggingface to achieve this
embedding = HuggingFaceEmbeddings()
def emb_text(text):
    try:
        embedding_vector = embedding.embed_query(text)
        return embedding_vector
    except Exception as e:
        print(f"have exception: {e}")
        return None

test_embedding = emb_text("This is a test")
# 这里输出的是两个内容：
# 1. embedding_dim: 表示embedding向量的维度大小
# 2. test_embedding[:10]: 表示测试embedding向量的前10个元素
embedding_dim = len(test_embedding)
# print(embedding_dim) # 768
# print(test_embedding[:10])


collection_name = "my_rag_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)


milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)


# res = get_completion_from_prompt("hello")
# print(res)

#  insert
data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

milvus_client.insert(collection_name=collection_name, data=data)



question = "How is data stored in milvus?"


# search
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        emb_text(question)
    ],  # Use the `emb_text` function to convert the question to an embedding vector
    limit=3,  # Return top 3 results
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text"],  # Return the text field
)

# distance 表示查询向量与检索到的向量之间的相似度距离
# 在这个例子中，我们使用的是内积（IP）作为距离度量方式
# 内积值越大表示向量越相似，值越小表示越不相似，最大理论上是1
retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))

# change to string
# [0]是拿到文本内容
context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)
# then, we do next thing
SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
USER_PROMPT = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

# llm 
messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]
res = get_completion_from_messages(messages)

print(res)










