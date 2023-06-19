import argparse
import pinecone
import numpy as np
from nomic import atlas
import os
import pandas as pd
from pprint import pprint

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument("--index", default="default", help="Pinecone Index")
parser.add_argument("--namespace", default=False, help="Pinecone Namespace")
parser.add_argument("--file", default="./output/default/contents.csv", help="Specify the path to the CSV")
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
df = pd.read_csv(args.file)
all_ids = df.iloc[:, 0].tolist()

ids = []
embeddings = []

# Fetch vectors for the IDs in batches of 100
for i in range(0, len(all_ids), 100):
    pprint("Fetching batch of IDs from {} to {}".format(i, i+100))
    batch_ids = all_ids[i:i+100]
    result = index.fetch(ids=batch_ids, **query_params)
    for id, vector in result['vectors'].items():
        ids.append(id)
        embeddings.append(vector['values'])

pprint("Fetched {} vectors".format(len(embeddings)))

embeddings = np.array(embeddings)

pprint("Pushing embeddings to Atlas")
atlas.map_embeddings(embeddings=embeddings, data=[{'id': id} for id in ids], id_field='id')
pprint("Done")
