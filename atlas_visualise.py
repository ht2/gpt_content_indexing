import argparse
import pinecone
import numpy as np
from nomic import atlas
import os
import pandas as pd
import openai
from datetime import datetime

from pprint import pprint

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument("--index", default="default", help="Pinecone Index")
parser.add_argument("--namespace", default=False, help="Pinecone Namespace")
parser.add_argument("--file", default="./output/default/contents.csv", help="Specify the path to the CSV")
parser.add_argument("--lines", type=int, help="Number of lines to read from the CSV")
args = parser.parse_args()

PINECONE_REGION = "us-east1-gcp"
PINECONE_DIMENSION_SIZE = 1536
PINECONE_METRIC = "euclidean"
PINECONE_POD_TYPE = "p1"

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=PINECONE_REGION)

index = pinecone.Index(args.index)

# Prepare query parameters
query_params = {}

# Conditionally add namespace to query_params
if args.namespace:
    query_params["namespace"] = args.namespace

# Load IDs from the CSV file
df = pd.read_csv(args.file, nrows=args.lines)
all_data = df.to_dict('records')

# Create a function to generate a brief description
def get_brief_description(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": text},
            {"role": "system", "content": "Describe this training in two to five words, followed by the URL to the content:"}
        ],
        temperature=0.5,
        max_tokens=100
    )

    return response['choices'][0]['message']['content'].strip()

ids = []
embeddings = []
descriptions = []

# Fetch vectors for the IDs in batches of 100
for i in range(0, len(all_data), 100):
    pprint(f"{datetime.now().strftime('%H:%M:%S')} - Fetching batch of IDs from {i} to {i+100}")
    batch_data = all_data[i:i+100]
    for d in batch_data:
        result = index.fetch(ids=[d['id']], **query_params)
        for id, vector in result['vectors'].items():
            ids.append(id)
            embeddings.append(vector['values'])
            description = get_brief_description(d['content'])
            descriptions.append(description)

pprint(f"{datetime.now().strftime('%H:%M:%S')} - Fetched {len(embeddings)} vectors")

embeddings = np.array(embeddings)

# prepare data for atlas.map_embeddings
data = [{'id': id, 'description': desc} for id, desc in zip(ids, descriptions)]

pprint(f"{datetime.now().strftime('%H:%M:%S')} - Pushing embeddings to Atlas")
atlas.map_embeddings(embeddings=embeddings, data=data, id_field='id')
pprint(f"{datetime.now().strftime('%H:%M:%S')} - Done")
