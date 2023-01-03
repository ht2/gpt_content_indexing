import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pandas as pd
import argparse
import numpy as np
import openai
import pprint
from transformers import GPT2TokenizerFast

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument with a flag and a name
parser.add_argument("--question", help="Specify the question you are asking")
parser.add_argument("--file", default="./output/indexed_content.csv", help="Specify the path to the CSV containing the content")
parser.add_argument("--embeddings", default="./output/embeddings.csv", help="Specify the path to the embeddings CSV")
parser.add_argument("--show_prompt", default=False, help="Output the prompt sent to OpenAI")
parser.add_argument("--allow_hallucinations", default=False, help="Don't restrict answers to be based from the provided context")
parser.add_argument("--use_fine_tune", default=False, help="Use the fine tuned model")
args = parser.parse_args()

MODEL_NAME = "curie"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"
COMPLETIONS_MODEL = "davinci:ft-learning-pool:strm-prompts-2022-12-20-18-07-34" if args.use_fine_tune else "text-davinci-003"
MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def load_embeddings(filename: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    filename is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(filename, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        title, heading = section_index
        content = document_section.content.replace("\n", " ");
            
        chosen_sections.append(f"{SEPARATOR}{title} - {heading} - {content}")
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    if args.show_prompt:
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))
    
    print("allow", args.allow_hallucinations)
    if args.allow_hallucinations == True:
        header = """Answer the question as truthfully as possible using the provided context. If the answer is not in the provided context, you may make a best guess using your wider knowledge."\n\nContext:\n"""
    else:
        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)
        exit()

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}


# Fetch the embeddings from the CSV
document_embeddings = load_embeddings(args.embeddings)

df = pd.read_csv(args.file)
df = df.set_index(["title", "heading"])
response = answer_query_with_context(args.question, df, document_embeddings, show_prompt=args.show_prompt)
print("")
print(f"Answer: {response}")