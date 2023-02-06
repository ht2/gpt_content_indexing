import argparse
from gpt_index import GPTPineconeIndex
import os
import pinecone

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument with a flag and a name
parser.add_argument("--question", help="Specify the question you are asking")
parser.add_argument("--slack", action=argparse.BooleanOptionalAction, help="Listen for slack messages and respond to them")
parser.add_argument("--dir", default="./output/default/", help="Specify the directory containing the index.json")
parser.add_argument("--index_name", default="default", help="Specify the pinecone directory we will query embeddings from")
args = parser.parse_args()

# Create the index
pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment="us-east1-gcp")
pinecone_index = pinecone.Index(args.index_name)

# Read and load the documents
contentDir = args.dir.rstrip("/")
contentsFile = f"{contentDir}/contents.json"
base_index = GPTPineconeIndex.load_from_disk(contentsFile, pinecone_index=pinecone_index)

if args.slack:
    print('Slack is not yet supported')
else:
    response = base_index.query(args.question)
    print(response)