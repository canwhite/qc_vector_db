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

# classic embedding 
def generate_embedding(video_url):
	"""
    Generate embeddings for a given video URL using the Twelve Labs API.

    This function creates an embedding task for the specified video URL using
    the Marengo-retrieval-2.6 engine. It monitors the task progress and waits
    for completion. Once done, it retrieves the task result and extracts the
    embeddings along with their associated metadata.

    Args:
        video_url (str): The URL of the video to generate embeddings for.

    Returns:
        tuple: A tuple containing two elements:
            1. list: A list of dictionaries, where each dictionary contains:
                - 'embedding': The embedding vector as a list of floats.
                - 'start_offset_sec': The start time of the segment in seconds.
                - 'end_offset_sec': The end time of the segment in seconds.
                - 'embedding_scope': The scope of the embedding (e.g., 'shot', 'scene').
            2. EmbeddingsTaskResult: The complete task result object from Twelve Labs API.

    Raises:
        Any exceptions raised by the Twelve Labs API during task creation,
        execution, or retrieval.
    """
