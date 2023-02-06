import pinecone
import os
from gpt_index import GPTPineconeIndex, SimpleDirectoryReader

# Create the index
pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment="us-east1-gcp")
index_name="default"
pinecone.create_index(index_name, dimension=1536, metric="euclidean", pod_type="p1")
pinecone_index = pinecone.Index(index_name)

# Read and load the documents
documents = SimpleDirectoryReader('../input/LXP').load_data()
index = GPTPineconeIndex(documents, pinecone_index=pinecone_index, chunk_size_limit=500)
index.save_to_disk('index.json')
