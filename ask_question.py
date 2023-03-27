import argparse
import datetime
import os
import json
import pandas as pd
import numpy as np
import openai
import pinecone
from pprint import pprint
import sys
import time

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument with a flag and a name
parser.add_argument("--question", help="Specify the question you are asking")
parser.add_argument("--dir", default="./output/default/", help="Specify the directory containing the contents.csv and embeddings.csv")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="Enable debug mode (show prompts and other info)")
parser.add_argument("--imagine", action=argparse.BooleanOptionalAction, help="Don't restrict answers to be based from the provided context")
parser.add_argument("--show_prompt", action=argparse.BooleanOptionalAction, help="Show the full prompt")
parser.add_argument("--stream", action=argparse.BooleanOptionalAction, help="Stream out the response")
parser.add_argument("--experiment_hyde", action=argparse.BooleanOptionalAction, help="Generate an answer from the question, and use that for embedding lookup (https://twitter.com/mathemagic1an/status/1615378778863157248/https://arxiv.org/pdf/2212.10496.pdf)")
parser.add_argument("--custom_instructions", default=False, help="Inject a custom set of instructions before the context")

parser.add_argument("--embedding_type", default="csv", choices=["csv", "pinecone"], help="Format to save embeddings in")
parser.add_argument("--pinecone_index", default="default", help="Pinecone Index")
parser.add_argument("--pinecone_namespace", default="content", help="Pinecone Namespace")
parser.add_argument("--pinecone_top_k", default=10, type=int, help="The number of results to return from the Pinecone index")

parser.add_argument("--completion_type", default="text", choices=["text", "chat"], help="Pinecone Namespace")
parser.add_argument("--text_model", default="text-davinci-003", help="The text completions model to use (defaults to text-davinci-003)")
parser.add_argument("--chat_model", default="gpt-4", help="The chat completions model to use (defaults to gpt-4)")
parser.add_argument("--max_tokens", default=600, type=int, help="The maximum number of tokens to generate (defaults to 600)")
parser.add_argument("--max_context", default=1000, type=int, help="The maximum length of content to include (defaults to 1000)")

args = parser.parse_args()

if args.completion_type == "text":
    COMPLETIONS_MODEL = args.text_model
elif args.completion_type == "chat":
    COMPLETIONS_MODEL = args.chat_model
else:
    raise ValueError("Invalid completion_type")

def timeprint(text):
    print(f"{datetime.datetime.now().time()}: {text}")

if args.debug:
    timeprint("Debug mode enabled.")
    timeprint(f"Completions mode: {args.completion_type}")
    timeprint(f"Model: {COMPLETIONS_MODEL}")
    timeprint(f"Max tokens: {args.max_tokens}")
    timeprint(f"Max context: {args.max_context}")
    timeprint(f"Imagining: {args.imagine}")
    timeprint(f"Experiment HyDE: {args.experiment_hyde}")
    if args.embedding_type == "pinecone":
        timeprint(f"Pinecone Index: {args.pinecone_index}")
        timeprint(f"Pinecone Namespace: {args.pinecone_namespace}")
        timeprint(f"Pinecone Top K: {args.pinecone_top_k}")
    

EMBEDDINGS_MODEL = "text-embedding-ada-002"
PINECONE_REGION="us-east1-gcp"
SEPARATOR = "\n* "
SEPARATOR_LEN = 3


def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_question_embedding(question: str) -> list[float]:
    if args.experiment_hyde:
        timeprint(f"Experimental HyDE mode activated. Fetching answer to Q: {question}")
        prompt = f"Answer this question as well as you can. The answer will be used to generate embeddings and search for related answers in a contextual knowledge bank for the original question. Question: {question} Answer:"
        response = openai.Completion.create(
            prompt=prompt,
            temperature= 0.0,
            max_tokens=args.max_tokens,
            model=COMPLETIONS_MODEL
        )
        answer = response["choices"][0]["text"].strip(" \n")
        timeprint(f"Using generated answer for embeddings... {answer}")
        return get_embedding(answer, EMBEDDINGS_MODEL)
    else:
        return get_embedding(question, EMBEDDINGS_MODEL)

def load_embeddings(filename: str) -> dict[tuple[str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    filename is the path to a CSV with exactly these named columns: 
        "id", "0", "1", ... up to the length of the embedding vectors.
    """
    
    timeprint(f"Loading embeddings from {filename}")
    df = pd.read_csv(filename, header=0)
    max_dim = max([int(c) for c in df.columns if c != "id"])
    return {
           (r.id): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))

def get_similarities_from_dict(content_embeddings:dict[(str, str), np.array], question_embedding: list[float]):
      document_similarities = sorted([
          (vector_similarity(question_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in content_embeddings.items()
      ], reverse=True)
      
      return document_similarities

def get_similarities_from_pinecone(index: pinecone.Index, question_embedding: list[float]):
    results = index.query(
      vector=question_embedding,
      top_k=args.pinecone_top_k,
      namespace=args.pinecone_namespace,
      include_values=False
    )
    document_similarities = [(match['score'], match['id']) for match in results['matches']]
    return document_similarities

def order_document_sections_by_question_similarity(question: str, embedding_type: str, content_embeddings) -> list[(float, (str, str))]:
    """
    Find the question embedding for the supplied question, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    question_embedding = get_question_embedding(question)
    if embedding_type == "csv":
        return get_similarities_from_dict(content_embeddings, question_embedding)
    elif embedding_type == "pinecone":
        return get_similarities_from_pinecone(index = content_embeddings, question_embedding=question_embedding)

def find_context(
        df: pd.DataFrame, 
        question:str,
        embedding_type,
        content_embeddings,
        ):
    """
    Find the most relevant sections of the document to answer the question.
    """
    most_relevant_document_sections = order_document_sections_by_question_similarity(question, embedding_type, content_embeddings)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + SEPARATOR_LEN
        if chosen_sections_len > args.max_context:
            if args.debug:
                timeprint(f"Max context reached {args.max_context}. {len(chosen_sections)} document sections selected")
            break

        content = document_section.content.replace("\n", " ");
        url = document_section.url;
            
        chosen_sections.append(f"{SEPARATOR}{content} (URL: {url})")
        chosen_sections_indexes.append(str(section_index))
    
    if len(chosen_sections) == 0:
        return "NO CONTEXT FOUND."
    # Useful diagnostic information
    if args.debug:
        timeprint(f"Selected {len(chosen_sections)} document sections")
        timeprint(f"Total content length: {chosen_sections_len}")

    return "".join(chosen_sections)


def get_system_instructions():
    prompt = "You are a helpful assistant who answers questions diligently for an end user."

    if args.custom_instructions:
        prompt+= args.custom_instructions
    else:
        prompt = "You are given a question and a context. The context is a collection of text that may contain the answer to the question. The context may also contain URLs that may be helpful in answering the question. Optimise your output for display on a command line.\n\n"
        if args.imagine:
            prompt+= "Answer the question using the provided context, but if the answer is not in the provided context, you may make your own guess. Explain the reasoning for your guess in your answer."
        else:
            prompt+= "If the answer is not contained within the context below, say 'I don't know.' Follow this with 'Here is additional information found whilst searching:'. Then include, a summary of each part of the Context, with its URL. Do NOT use anything but the provided context to answer the question. Do not source information from your wider knowledge.\n\n"
            prompt+= "Answer the question as truthfully as possible using ONLY the provided context. You should use as much detail from the given context as possible when answering the question.\n\n"
            prompt+= "Within the context are URLs relavant to the content, when an answer is found, print the URLs used to form the answer."
    return prompt


def call_text_completion(question, context):
    """
    Call the OpenAI Text completion API to generate an answer to the question.
    """
    prompt = get_system_instructions()
    prompt+= "\n\n"
    prompt+= f"Context:\n{context}"
    prompt+= "\n\n"
    prompt+= f"Question: {question}"

    if args.show_prompt:
        timeprint(f"Prompt: {prompt}")
    
    try:
        return openai.Completion.create(
            prompt=prompt,
            temperature= 1.0 if args.imagine else 0.0,
            max_tokens=args.max_tokens,
            model=COMPLETIONS_MODEL,
            stream=args.stream
        )
    except Exception as e:
        print("Error: ", e)
        exit(1)


def call_chat_completion(question, context):
    """
    Call the OpenAI Chat Completion API to generate an answer to the question.
    """
    system_role_content = get_system_instructions()

    if args.custom_instructions:
        user_role_content = args.custom_instructions
    elif args.imagine:
        user_role_content = "Answer the question using the provided context, but if the answer is not in the provided context, you may make your own guess. Explain the reasoning for your guess in your answer."
    else:
        user_role_content = "Answer the following question using ONLY the provided context. You should use as much detail from the given context as possible when answering the question.\n\n"
    user_role_content+= f"Context:\n{context}"
    user_role_content+= "\n\n"
    user_role_content+= f"Question: {question}"

    if args.show_prompt:
        timeprint(f"System Role Content: {system_role_content}")
        timeprint(f"User Role Content: {user_role_content}")
    
    try:
        return openai.ChatCompletion.create(
            model=COMPLETIONS_MODEL,
            messages=[
                {"role":"system", "content": system_role_content},
                {"role":"user", "content": user_role_content}
            ],
            temperature= 1.0 if args.imagine else 0.0,
            max_tokens=args.max_tokens,
            stream=args.stream
        )
    except Exception as e:
        print("Error: ", e)
        exit(1)

def answer_question(
    question:str,
    context:str
) -> str:
    if args.debug:
        timeprint(f"Using {args.completion_type} completion type")

    if args.completion_type == "text":
        response = call_text_completion(question, context)
    elif args.completion_type == "chat":
        response = call_chat_completion(question, context)
    else:
        timeprint("Invalid completion type")
        exit(1)

    print(f"\nQuestion: {question}\n")
    if (args.stream == True):
        sys.stdout.write('Answer: ')
        sys.stdout.flush()
        for chunk in response:
            if args.completion_type == "text":
                sys.stdout.write(chunk.choices[0]['text'])
                sys.stdout.flush()
            else:
                # Stream out the generated text for each completion
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta', {})
                    content = delta.get('content', '')
                    sys.stdout.write(content)
                    sys.stdout.flush()
        sys.stdout.write('\n\n')
    else:
        if args.completion_type == "text":
            print("Answer: ", response['choices'][0]['text'])
        else:
            print("Answer: ", response['choices'][0]['message']['content'])


def main():
    contentDir = args.dir.rstrip("/")
    contentsFile = f"{contentDir}/contents.csv"

    if not os.path.exists(contentsFile):
        timeprint("Error: contents.csv must exist in the provided directory")
        sys.exit()

    timeprint(f"Load contents.csv...")
    df = pd.read_csv(contentsFile)
    timeprint(f"Loaded!")
    df = df.set_index(["id"])

    embedding_type = args.embedding_type
    if embedding_type == "csv":
      # File based embeddings - note these load into memory and can be slow
      embeddingsFile = f"{contentDir}/embeddings.csv"
      print(f"Loading embeddings from {embeddingsFile}...")

      if not os.path.exists(contentsFile) or not os.path.exists(embeddingsFile):
          print("Error: embeddings.csv must exist in the provided directory")
          sys.exit()

      start_time = time.time()
      # Fetch the embeddings from the CSV
      content_embeddings = load_embeddings(embeddingsFile)
      load_time = time.time() - start_time
      print(f"Embeddings loaded in {round(load_time,2)} seconds")
    elif embedding_type == "pinecone":
      # Use a Pinecone index
      pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=PINECONE_REGION)
      content_embeddings = pinecone.Index(args.pinecone_index)

    timeprint(f"Answering question...")

    # Clean the question of any prompts
    question = args.question.strip()
    if question.endswith("?") == False:
        question = question + "?"

    
    # Get and return the answer
    start_time = time.time()
    context = find_context(question=question, df=df, embedding_type=embedding_type, content_embeddings=content_embeddings)
    
    answer_question(
        question=question,
        context=context
    )
    execution_time = (time.time() - start_time) * 1000
    timeprint(f"Answered in {round(execution_time,2)} ms")
    exit()

if __name__ == "__main__":
    timeprint(f"Entry point")
    main()