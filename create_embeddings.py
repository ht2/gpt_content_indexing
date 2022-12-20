import os
import argparse
import openai
import pprint
import pandas as pd
from transformers import GPT2TokenizerFast

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument with a flag and a name
parser.add_argument("--file", default="./output/indexed_content.csv", help="Specify the path to the CSV")
parser.add_argument("--out", default="embeddings", help="Specify the filename to save the embeddings")

args = parser.parse_args()
file = args.file;

df = pd.read_csv(file)
df = df.set_index(["title", "heading"])
print(f"{len(df)} rows in the data.")
sample = df.sample(5)
print("Sample (5 rows)", sample)


MODEL_NAME = "curie"
DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }

# Generate the embeddings
context_embeddings = compute_doc_embeddings(df)

# Convert the context_embeddings dictionary to a list of tuples, where each tuple is of the form (title, heading, embedding)
context_embeddings_list = [(k[0], k[1]) + tuple(v) for k, v in context_embeddings.items()]

# Create a DataFrame from the list of tuples
column_names = ['title', 'heading'] + [i for i in range(len(list(context_embeddings.values())[0]))]
df = pd.DataFrame(context_embeddings_list, columns=column_names)

# Save the DataFrame to a CSV file
dir = './output/';
filename = args.out + '.csv'
fullpath = dir + filename
df.to_csv(fullpath, index=False)
print("Done! Saved to " + fullpath)
