import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pandas as pd
import argparse
import numpy as np
import openai
from pprint import pprint
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

show_prompt = bool(args.show_prompt)
allow_hallucinations = bool(args.allow_hallucinations)

QUERY_EMBEDDINGS_MODEL = "text-embedding-ada-002"
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
        url = document_section.url;
            
        chosen_sections.append(f"{SEPARATOR}{title} - {heading} - {content} (URL: {url})")
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    if show_prompt:
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))
    

    if bool(allow_hallucinations) == True:
        print("Halluncinations are enabled!")
        header = "Answer the question based on the provided context. If the answer is not in the provided context, you may make a best guess using your wider knowledge."
        header += "The context provided contains multiple sections of text from a knowledge base and a URL for each. For each section of text (which starts with a \"*\" character), return a unique answer followed by the text 'More info:' followed by the URL. You may return up to three answers, each separated with two line breaks."
    else:
        header = "Answer the question as truthfully as possible using the provided context. You should use as much detail from the given context as possible when answering the question."
        header += "If the answer is not contained within the text below, say 'I don't know.' followed by the all the text in the 'Context' section (preceeded by 'Here is the closest information I could find to your question\\n\\n:'). "
        header += "Within the context are URLs. If an answer if found within a relevant section, return the answer and then three line breaks and then the text 'More info:' followed by the URL."
    
    header += ""

    header += "\n\nContext:\n"    
    header += "".join(chosen_sections) + "\n\n"
    header += "Q: " + question + "\n A:"

    return header
     

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
    
    print("\n\n")
    if show_prompt:
        print(prompt)
    else:
        print(f"Question: {query}")

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 1.0 if allow_hallucinations else 0.0,
    "max_tokens": 600,
    "model": COMPLETIONS_MODEL,
}


# Fetch the embeddings from the CSV
document_embeddings = load_embeddings(args.embeddings)

df = pd.read_csv(args.file)
df = df.set_index(["title", "heading"])
response = answer_query_with_context(args.question, df, document_embeddings, show_prompt=show_prompt)
print("")
print(f"Answer: {response}")