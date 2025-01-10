from pymilvus import MilvusClient,DataType
import sys
import os
from pathlib import Path

root_path = Path(__file__).parent.parent
root_path = str(root_path)
sys.path.append(root_path)

from llm import get_completion_from_messages,get_completion_from_prompt
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import textwrap

rec_path = Path(__file__).parent.parent / "data/milvus_rec.db"
rec_path = str(rec_path)
print(rec_path)


# 一些constant data
COLLECTION_NAME = "movie_search"
DIMENSION = 768
BATCH_SIZE = 1000

# Connect to Milvus Database
client = MilvusClient(rec_path)

if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)


# createSchema
# 1. Create schema
schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=False,
)

# 2. Add fields to schema
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=64000)
schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=64000)
schema.add_field(field_name="release_year", datatype=DataType.INT64)
schema.add_field(field_name="rating", datatype=DataType.VARCHAR, max_length=64000)
schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=64000)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)

# 3. Create collection with the schema
client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

# --- add index ---

# 1. Prepare index parameters
index_params = client.prepare_index_params()

# 2. Add an index on the embedding field
index_params.add_index(
    field_name="embedding", metric_type="IP", index_type="AUTOINDEX", params={}
)

# 3. Create index
client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)

# 4. Load Collection
client.load_collection(collection_name=COLLECTION_NAME)

# add dataset
dataset = load_dataset("hugginglearners/netflix-shows", split="train")


embedding = HuggingFaceEmbeddings()
def emb_texts(texts):
    """
    批量处理文本嵌入
    Args:
        texts (list): 需要嵌入的文本列表
    
    Returns:
        list: 返回每个文本对应的嵌入向量列表
    """
    try:
        # 使用embedding的embed_documents方法进行批量处理
        embedding_vectors = embedding.embed_documents(texts)
        return embedding_vectors
    except Exception as e:
        print(f"批量处理嵌入时发生异常: {e}")
        
        return [None] * len(texts)  # 返回与输入长度相同的None列表



# batch (data to be inserted) is a list of dictionaries
batch = []

# Embed and insert in batches
for i in tqdm(range(0, len(dataset))):

    # batch append item
    batch.append(
        {
            "title": dataset[i]["title"] or "",
            "type": dataset[i]["type"] or "",
            "release_year": dataset[i]["release_year"] or -1,
            "rating": dataset[i]["rating"] or "",
            "description": dataset[i]["description"] or "",
        }
    )
    
    # 这个判断用于检查是否达到批量插入的条件：
    # 1. 当batch中的记录数达到BATCH_SIZE时，执行批量插入
    # 2. 或者当处理到最后一条记录时（i == len(dataset) - 1），执行批量插入
    # 这样可以避免内存中积累过多数据，同时确保最后一批数据也能被插入
    if len(batch) % BATCH_SIZE == 0 or i == len(dataset) - 1:
        # item = dataset[i]
        embeddings = emb_texts([item["description"] for item in batch])
    
        # zip合并成一个元组
        for item, emb in zip(batch, embeddings):
            item["embedding"] = emb

        client.insert(collection_name=COLLECTION_NAME, data=batch)
        batch = []


def query(query, top_k=5):
    text, expr = query

    res = client.search(
        collection_name=COLLECTION_NAME,
        data=emb_texts(text),
        filter=expr,
        limit=top_k,
        output_fields=["title", "type", "release_year", "rating", "description"],
        search_params={
            "metric_type": "IP",
            "params": {},
        },
    )

    print("Description:", text, "Expression:", expr)

    for hit_group in res:
        print("Results:")
        for rank, hit in enumerate(hit_group, start=1):
            entity = hit["entity"]

            print(
                f"\tRank: {rank} Score: {hit['distance']:} Title: {entity.get('title', '')}"
            )
            print(
                f"\t\tType: {entity.get('type', '')} "
                f"Release Year: {entity.get('release_year', '')} "
                f"Rating: {entity.get('rating', '')}"
            )
            description = entity.get("description", "")
            print(textwrap.fill(description, width=88))
            print()



my_query = ("movie about a fluffly animal", 'release_year < 2019 and rating like "PG%"')

query(my_query)
