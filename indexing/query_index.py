import pinecone
import os
from gpt_index import GPTPineconeIndex, SimpleDirectoryReader
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.agents import initialize_agent

from gpt_index import GPTSimpleVectorIndex

# Create the index
pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment="us-east1-gcp")
index_name="jm-playground"
pinecone_index = pinecone.Index(index_name)

# Read and load the documents
base_index = GPTPineconeIndex.load_from_disk('./index.bin', pinecone_index=pinecone_index)
response = base_index.query("What versions of SCORM do we support?")
print(response)