import os
import pinecone
import argparse
import openai
import pandas as pd
from pprint import pprint

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument with a flag and a name
parser.add_argument("--file", default="./output/default/contents.csv", help="Specify the path to the CSV")
parser.add_argument("--embedding_type", default="pinecone", choices=["csv", "pinecone"], help="Format to save embeddings in")
parser.add_argument("--out", default="./output/default/embeddings.csv", help="Specify the filename to save the embeddings")
parser.add_argument("--pinecone_mode", default="replace", choices=["upsert", "replace"], help="Specify the mode to upsert or replace embeddings in Pinecone index")
parser.add_argument("--pinecone_index", default="default", help="Pinecone Index")
parser.add_argument("--pinecone_namespace", default="content", help="Pinecone Namespace")
args = parser.parse_args()

DOC_EMBEDDINGS_MODEL = "text-embedding-ada-002"
EMBEDDING_BATCH_SIZE = 1000
PINECONE_REGION="us-east1-gcp"
PINECONE_DIMENSION_SIZE=1536
PINECONE_METRIC="euclidean"
PINECONE_POD_TYPE="p1"
PINECONE_BATCH_SIZE = 100

def load_content_dataframe(filename):
    df = pd.read_csv(filename)
    df = df.set_index(["id"])
    print(f"{len(df)} rows in the data.")

    # drop rows with empty content
    df = df.dropna(subset=["content"])
    print(f"{len(df)} rows with content.")

    sampleSize = 5 if len(df) >=5 else len(df)
    sample = df.sample(sampleSize)
    print("Sample (5 rows)", sample)
    return df

def get_embedding(text: str, model: str) -> list[float]:
    return openai.Embedding.create(
      model=model,
      input=text
    )

def get_doc_embedding(text: str) -> list[float]:
    result = get_embedding(text, DOC_EMBEDDINGS_MODEL)
    return result["data"][0]["embedding"]

def compute_doc_embeddings_old(df: pd.DataFrame) -> dict[tuple[str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    print('Generating the embeddings...')
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) if isinstance(r.content, str) else ""
        for idx, r in df.iterrows()
    }

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    print('Generating the embeddings...')
    results = {}
    batch_size = EMBEDDING_BATCH_SIZE
    processed = 0
    total = len(df)
    for i in range(0, total, batch_size):
        batch_rows = df[i:i+batch_size]
        batch_texts = [r.content.replace("\n", " ") if isinstance(r.content, str) and r.content != "" else "" for _, r in batch_rows.iterrows()]
        batch_embeddings = get_embedding(batch_texts, DOC_EMBEDDINGS_MODEL)
        for idx, embedding in zip(batch_rows.index, batch_embeddings["data"]):
            results[idx] = embedding["embedding"]
        processed += batch_size
        print(f"Processed embeddings for {processed}/{total} rows")
    return results

# Function to generate CSV from dataframe
def generate_csv_embeddings(embeddings_dict: dict[tuple[str], list[float]]):
    filename = args.out
    print('Saving file to CSV...')

    # Convert the context_embeddings dictionary to a list of tuples, where each tuple is of the form (id, embedding)
    context_embeddings_list = [(k,) + tuple(v)
                               for k, v in embeddings_dict.items()]

    # Create a DataFrame from the list of tuples
    column_names = ['id'] + [i for i in range(len(list(embeddings_dict.values())[0]))]
    df = pd.DataFrame(context_embeddings_list, columns=column_names)

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Done! Saved to {filename}")

def find_or_create_index(index_name:str, namespace:str, recreate_index:bool=False):
    try:
        # Check for index
        pinecone.describe_index(index_name)
        print(f"Index `{index_name}` found...")
    except pinecone.core.client.exceptions.NotFoundException as e:
        # Index was not found, create it
        print(f"Index `{index_name}` does not exist, creating...")
        pinecone.create_index(index_name, dimension=PINECONE_DIMENSION_SIZE, metric=PINECONE_METRIC, pod_type=PINECONE_POD_TYPE)
        print(f"Index created!")

    # Get the index    
    index = pinecone.Index(index_name)
    
    # If the index has been found, check if we need to remake it
    if recreate_index:
        print('Deleting vectors in namespace...')
        index.delete(delete_all=True, namespace=namespace)
        print(f"Done")
    return index

def insert_vectors(index, vectors, mode):
    if mode == "upsert":
        index.upsert(vectors)
    elif mode == "replace":
        index.replace(vectors)
    else:
        raise ValueError("Invalid value for --mode argument, must be 'upsert' or 'replace'")

# Function to upsert the embeddings to Pinecone
def generate_pinecone_embeddings(embeddings_dict:dict[tuple[str], list[float]]):
    index_name = args.pinecone_index
    namespace = args.pinecone_namespace
    recreate_index = args.pinecone_mode == "replace"

    print(f"Generating Pinecone embeddings for index:{index_name}...")
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=PINECONE_REGION)
    index = find_or_create_index(index_name, namespace, recreate_index)

    print('Inserting vectors...')
    vectors = []
    for i, (id, embedding) in enumerate(embeddings_dict.items()):
        vectors.append((id, embedding, {}))
        if (i + 1) % PINECONE_BATCH_SIZE == 0:
            index.upsert(vectors=vectors, namespace=namespace)
            vectors = []
    if vectors:
            index.upsert(vectors=vectors, namespace=namespace)

def main():
    content_df = load_content_dataframe(args.file)

    # Generate the embeddings
    embeddings_dict = compute_doc_embeddings(df=content_df)

    if args.embedding_type == 'csv':
        generate_csv_embeddings(embeddings_dict=embeddings_dict)
    elif args.embedding_type == 'pinecone':
        generate_pinecone_embeddings(embeddings_dict=embeddings_dict)

    print('All done!')
# Entry point
if __name__ == "__main__":
    main()