import argparse
from gpt_index import GPTPineconeIndex, SimpleDirectoryReader
import os
import pinecone
from pprint import pprint


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument with a flag and a name
parser.add_argument("--input_dir", default="./input/", help="Specify the directory to write the contents.json to")
parser.add_argument("--output_dir", default="./output/default/", help="Specify the directory to write the contents.json to")
parser.add_argument("--index_name", default="default", help="Specify the pinecone directory we will query embeddings from")
args = parser.parse_args()

index_name = args.index_name
# Find (and delete) or create the index
pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment="us-east1-gcp")
try:
  pinecone_index = pinecone.describe_index(index_name)
  print('Index found, deleting and recreating...')
  pinecone.delete_index(index_name)
except pinecone.core.client.exceptions.NotFoundException as e:
  print('Index does not exist, creating...')

pinecone.create_index(index_name, dimension=1536, metric="euclidean", pod_type="p1")
pinecone_index = pinecone.Index(index_name)

# Read and load the documents
documents = SimpleDirectoryReader(input_dir=args.input_dir, recursive=True).load_data()
index = GPTPineconeIndex(documents, pinecone_index=pinecone_index)

outputDir = args.output_dir.rstrip("/")
contentsFile = f"{outputDir}/contents.json"
index.save_to_disk(contentsFile)