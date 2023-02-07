import argparse
import datetime
import os
import pandas as pd
import numpy as np
import openai
import pinecone
from pprint import pprint
import slack
import sys
import time
from transformers import GPT2TokenizerFast

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument with a flag and a name
parser.add_argument("--question", help="Specify the question you are asking")
parser.add_argument("--slack", action=argparse.BooleanOptionalAction, help="Listen for slack messages and respond to them")
parser.add_argument("--dir", default="./output/default/", help="Specify the directory containing the contents.csv and embeddings.csv")
parser.add_argument("--show_prompt", action=argparse.BooleanOptionalAction, help="Output the prompt sent to OpenAI")
parser.add_argument("--imagine", action=argparse.BooleanOptionalAction, help="Don't restrict answers to be based from the provided context")
parser.add_argument("--custom_model", default=False, help="Use the fine tuned model")
parser.add_argument("--stream", action=argparse.BooleanOptionalAction, help="Stream out the response")
parser.add_argument("--custom_prompt", default=False, help="Inject a custom prompt infront of the context")

parser.add_argument("--embedding_type", default="csv", choices=["csv", "pinecone"], help="Format to save embeddings in")
parser.add_argument("--pinecone_index", default="default", help="Pinecone Index")
parser.add_argument("--pinecone_namespace", default="content", help="Pinecone Namespace")
args = parser.parse_args()

COMPLETIONS_MODEL = args.custom_model if args.custom_model else "text-davinci-003"
EMBEDDINGS_MODEL = "text-embedding-ada-002"
PINECONE_REGION="us-east1-gcp"
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

def get_question_embedding(text: str) -> list[float]:
    return get_embedding(text, EMBEDDINGS_MODEL)

def load_embeddings(filename: str) -> dict[tuple[str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    filename is the path to a CSV with exactly these named columns: 
        "id", "0", "1", ... up to the length of the embedding vectors.
    """
    
    print(f"Loading embeddings from {filename}")
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
      top_k=6,
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

def construct_prompt(
  question: str,
  embedding_type,
  content_embeddings,
  df: pd.DataFrame,
  imagine: bool
  ) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_question_similarity(question, embedding_type, content_embeddings)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        id = section_index
        content = document_section.content.replace("\n", " ");
        url = document_section.url;
            
        chosen_sections.append(f"{SEPARATOR}{id} - {content} (URL: {url})")
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    if args.show_prompt:
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))
    
    if args.custom_prompt:
        header = args.custom_prompt
    else:
        # Provide a looser prompt to allow the system to invent  answers outside of the context
        if imagine:
            print("Halluncinations are enabled!")
            header = "Answer the question using the provided context, but if the answer is not in the provided context, you may make your own guess. Explain the reasoning for your guess in your answer."
        else:
            header = "Answer the question as truthfully as possible using the provided context. You should use as much detail from the given context as possible when answering the question."
            header += "If the answer is not contained within the text below, say 'I don't know.' followed by the all the text in the 'Context' section, with their respective URLs (preceeded by 'Here is the closest information I could find to your question\\n\\n:'). "
            header += "Within the context are URLs. If an answer if found within a relevant section, return the answer and then three line breaks and then the text 'More info:' followed by all the relevant URLs provided for the context, using bullet points to separate them."
        
    header += "\n\nContext:\n"    
    header += "".join(chosen_sections) + "\n\n"
    header += "Q: " + question

    return header
     
def answer_question_with_context(
    question: str,
    df: pd.DataFrame,
    embedding_type: str,
    content_embeddings,
    show_prompt: bool = False,
    print_question: bool = True,
    return_answer: bool = False,
    imagine = False,
) -> str:
    answer = ""
    prompt = construct_prompt(
        question,
        embedding_type,
        content_embeddings,
        df,
        imagine
    )
    
    if show_prompt:
        if return_answer:
            answer += f"{prompt}\n\n"
        else:
            print(f"\n\n{prompt}")
        
    if(print_question):
        print(f"\n\n{datetime.datetime.now().time()}: Question: {question}")

    print(f"{datetime.datetime.now().time()}: Sending Q to OpenAI...")
    response = openai.Completion.create(
        prompt=prompt,
        temperature= 1.0 if imagine else 0.0,
        max_tokens=600,
        model=COMPLETIONS_MODEL,
        stream=args.stream
    )

    if (return_answer):
        answer += response["choices"][0]["text"].strip(" \n")
        return f"Answer: {answer}"

    # Stream the response to the user
    if (args.stream == True):
        sys.stdout.write('Answer: ')
        sys.stdout.flush()
        for token in response:
            sys.stdout.write(token.choices[0]['text'])
            sys.stdout.flush()
    else:
        answer = response["choices"][0]["text"].strip(" \n")
        print(f"{datetime.datetime.now().time()}: Answer: {answer}")

def main():
    contentDir = args.dir.rstrip("/")
    contentsFile = f"{contentDir}/contents.csv"

    if not os.path.exists(contentsFile):
        print("Error: contents.csv must exist in the provided directory")
        sys.exit()

    df = pd.read_csv(contentsFile)
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

    # If we are looking to listen for Slack...
    if (args.slack):    
        print("Listening for Slack messages...")
        client = slack.WebClient(os.environ.get('SLACK_BOT_API_KEY'))
        slack_token = os.environ.get('SLACK_BOT_API_KEY')

        # The ID of the bot that when mentioned, we want to respond to
        bot_id = os.environ.get('SLACK_BOT_ID')

        # Listen for messages in Slack
        @slack.RTMClient.run_on(event='message')
        def respond_to_message(**payload):
            try:
                data = payload['data']

                # Check if the bot was mentioned and ensure it is not the bot talking to itself!
                if 'bot_id' not in data and bot_id in data.get('text', ''):
                    current_time = datetime.datetime.now().time()
                    # Extract the username from the returned JSON object
                    user_info = client.users_info(user=data['user'])
                    username = user_info['user']['name']

                    # Get the user's information from the Slack API
                    user_question = data['text']
                    thread_ts = data.get("ts")

                    print("------")
                    print(f"{datetime.datetime.now().time()}: User {username} asks \"{user_question}\"")
                    channel_id = data['channel']
                    # Allow some hallucinations if the CLI or user wants it
                    imagine = args.imagine or '[--imagine]' in data.get('text', '')
                    show_prompt = args.show_prompt or '[--show_prompt]' in data.get('text', '')
                    
                    holding_message = "Let me look that up for you! This might take a few seconds..."
                    if imagine:
                        holding_message += " (Imagine mode enabled!)"
                    if show_prompt:
                        holding_message += " (Prompt enabled!)"

                    # Return a holding message back to the Slack thread
                    client.chat_postMessage(
                        channel=channel_id,
                        text=holding_message,
                        as_user=True,
                        thread_ts=thread_ts
                    )

                    start_time = time.time()
                    answer = answer_question_with_context(
                        question=user_question,
                        df=df,
                        embedding_type=embedding_type,
                        content_embeddings=content_embeddings,
                        show_prompt=show_prompt,
                        print_question=False,
                        return_answer=True,
                        imagine=imagine
                    )
                    execution_time = (time.time() - start_time) * 1000
                    print(f"{datetime.datetime.now().time()}: Responded in {round(execution_time)}ms")

                    answer += f"\n\n_I took {round(execution_time/1000,2)} seconds to respond_"

                    # Return the answer back to the Slack channel
                    client.chat_postMessage(
                        channel=channel_id,
                        text=answer,
                        as_user=True,
                        thread_ts=thread_ts
                    )
            except Exception as e:
                import traceback
                print("exception caught:", e)
                traceback.print_exc()
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"Line number of error: {exc_tb.tb_lineno}, in file {fname}")

                # Return the answer back to the Slack channel
                client.chat_postMessage(
                    channel=channel_id,
                    text=f"Sorry, something went wrong!\n\n{type(e).__name__} {e}",
                    as_user=True,
                    thread_ts=thread_ts
                )

        # Start the slack client
        rtm_client = slack.RTMClient(token = slack_token)
        rtm_client.start()
    else:
        print(f"{datetime.datetime.now().time()}: Answering question...")
        answer_question_with_context(
            question=args.question,
            df=df,
            embedding_type=embedding_type,
            content_embeddings=content_embeddings,
            show_prompt=args.show_prompt,
            print_question=True,
            return_answer=False,
            imagine=args.imagine
        )
        exit()

if __name__ == "__main__":
    main()