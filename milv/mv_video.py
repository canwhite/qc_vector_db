# video similarity 
# twelvelabs，专注于视频理解领域。该工具的目标用户主要是开发者和产品经理，旨在通过先进的AI技术，帮助用户从视频中提取信息，理解视频内容。
import sys
from pathlib import Path
# init milvus
from pymilvus import MilvusClient
# 先让拿到
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)


video_db_path = str(Path(__file__).parent.parent / "data/milvus_twelvelabs_demo.db")
print(video_db_path)
import os 
from dotenv import load_dotenv
load_dotenv()

from twelvelabs import TwelveLabs
from twelvelabs.models.embed import EmbeddingsTask

# Initialize the Milvus client
milvus_client = MilvusClient(video_db_path)

print("Successfully connected to Milvus")


# ---create collection 
# Initialize the collection name
collection_name = "twelvelabs_demo_collection"

# Check if the collection already exists and drop it if it does
if milvus_client.has_collection(collection_name=collection_name):
    milvus_client.drop_collection(collection_name=collection_name)

# Create the collection
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=1024  # The dimension of the Twelve Labs embeddings, same as 
)

print(f"Collection '{collection_name}' created successfully")

TWELVE_LABS_API_KEY = os.getenv('TWELVE_LABS_API_KEY')
print(TWELVE_LABS_API_KEY)


# create twelvelib client
twelvelabs_client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)

# --- get embedding results ---
def generate_embedding(video_url):
    """
    Generate embeddings for a given video URL using the Twelve Labs API.
    ... 文档注释保持不变 ...
    """
    # 创建embedding任务
    task = twelvelabs_client.embed.task.create(
        model_name="Marengo-retrieval-2.7",
        video_url=video_url
    )

    print(f"Created task: id={task.id} engine_name={task.engine_name} status={task.status}")
    
    # 定义回调函数来监控任务进度
    def on_task_update(task: EmbeddingsTask):
        print(f"  Status={task.status}")

    # 等待任务完成
    status = task.wait_for_done(
        sleep_interval=2,
        callback=on_task_update
    )
    print(f"Embedding done: {status}")

    # 获取任务结果
    task_result = twelvelabs_client.embed.task.retrieve(task.id)

    # 提取并返回embeddings
    embeddings = []
    for v in task_result.video_embeddings:
        embeddings.append({
            'embedding': v.embedding.float,
            'start_offset_sec': v.start_offset_sec,
            'end_offset_sec': v.end_offset_sec,
            'embedding_scope': v.embedding_scope
        })
    
    return embeddings, task_result



video_url = "https://www.youtube.com/watch?v=rI-tjzfCsX8&ab_channel=RockZhang"
# Assuming this function exists from previous step
embeddings, task_result = generate_embedding(video_url)
print(f"Generated {len(embeddings)} embeddings for the video")
for i, emb in enumerate(embeddings):
    print(f"Embedding {i+1}:")
    print(f"  Scope: {emb['embedding_scope']}")
    print(f"  Time range: {emb['start_offset_sec']} - {emb['end_offset_sec']} seconds")
    print(f"  Embedding vector (first 5 values): {emb['embedding'][:5]}")
    print()


# --Inserting Embeddings into Milvus--
def insert_embeddings(milvus_client, collection_name, task_result, video_url):
    """
    Insert embeddings into the Milvus collection.

    Args:
        milvus_client: The Milvus client instance.
        collection_name (str): The name of the Milvus collection to insert into.
        task_result (EmbeddingsTaskResult): The task result containing video embeddings.
        video_url (str): The URL of the video associated with the embeddings.

    Returns:
        MutationResult: The result of the insert operation.

    This function takes the video embeddings from the task result and inserts them
    into the specified Milvus collection. Each embedding is stored with additional
    metadata including its scope, start and end times, and the associated video URL.
    """
    data = []

    for i, v in enumerate(task_result.video_embeddings):
        data.append({
            "id": i,
            "vector": v.embedding.float,
            "embedding_scope": v.embedding_scope,
            "start_offset_sec": v.start_offset_sec,
            "end_offset_sec": v.end_offset_sec,
            "video_url": video_url
        })

    insert_result = milvus_client.insert(collection_name=collection_name, data=data)
    print(f"Inserted {len(data)} embeddings into Milvus")
    return insert_result


# Insert embeddings into the Milvus collection
insert_result = insert_embeddings(milvus_client, collection_name, task_result, video_url)
print(insert_result)


# ---Similarity Search---
def perform_similarity_search(milvus_client, collection_name, query_vector, limit=5):
    """
    Perform a similarity search on the Milvus collection.

    Args:
        milvus_client: The Milvus client instance.
        collection_name (str): The name of the Milvus collection to search in.
        query_vector (list): The query vector to search for similar embeddings.
        limit (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        list: A list of search results, where each result is a dictionary containing
              the matched entity's metadata and similarity score.

    This function searches the specified Milvus collection for embeddings similar to
    the given query vector. It returns the top matching results, including metadata
    such as the embedding scope, time range, and associated video URL for each match.
    """
    search_results = milvus_client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=limit,
        output_fields=["embedding_scope", "start_offset_sec", "end_offset_sec", "video_url"]
    )

    return search_results

 
query_vector = task_result.video_embeddings[0].embedding.float

# Perform a similarity search on the Milvus collection
search_results = perform_similarity_search(milvus_client, collection_name, query_vector)

print("Search Results:")
for i, result in enumerate(search_results[0]):
    print(f"Result {i+1}:")
    print(f"  Video URL: {result['entity']['video_url']}")
    print(f"  Time Range: {result['entity']['start_offset_sec']} - {result['entity']['end_offset_sec']} seconds")
    print(f"  Similarity Score: {result['distance']}")
    print()

