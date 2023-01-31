import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import os
import slack
import time
import pandas as pd
import argparse
import numpy as np
import openai
from transformers import GPT2TokenizerFast

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument with a flag and a name
parser.add_argument("--question", help="Specify the question you are asking")
parser.add_argument("--slack", action=argparse.BooleanOptionalAction, help="Listen for slack messages and respond to them")
parser.add_argument("--dir", default="./output/default/", help="Specify the directory containing the contents.csv and embeddings.csv")
parser.add_argument("--show_prompt", action=argparse.BooleanOptionalAction, help="Output the prompt sent to OpenAI")
parser.add_argument("--imagine", action=argparse.BooleanOptionalAction, help="Don't restrict answers to be based from the provided context")
parser.add_argument("--use_fine_tune", action=argparse.BooleanOptionalAction, help="Use the fine tuned model")
parser.add_argument("--stream", action=argparse.BooleanOptionalAction, help="Stream out the response")
args = parser.parse_args()

QUESTION_EMBEDDINGS_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "

COMPLETIONS_MODEL = "davinci:ft-learning-pool:strm-prompts-2022-12-20-18-07-34" if args.use_fine_tune else "text-davinci-003"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))



def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_question_embedding(text: str) -> list[float]:
    return get_embedding(text, QUESTION_EMBEDDINGS_MODEL)

def load_embeddings(filename: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    filename is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    print(f"Loading embeddings from {filename}")
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

def order_document_sections_by_question_similarity(question: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the question embedding for the supplied question, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    question_embedding = get_question_embedding(question)
    
    document_similarities = sorted([
        (vector_similarity(question_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame, imagine: bool) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_question_similarity(question, context_embeddings)
    
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
    if args.show_prompt:
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))
    
    # Provide a looser prompt to allow the system to invent  answers outside of the context
    if imagine:
        print("Halluncinations are enabled!")
        header = "Answer the question using the provided context, but if the answer is not in the provided context, you may make your own guess. Explain the reasoning for your guess in your answer."
        header += "The context provided contains multiple sections of text from a knowledge base and a URL for each. For each section of text (which starts with a \"*\" character), return a unique answer followed by the text 'More info:' followed by the URL. You may return up to three answers, each separated with two line breaks."
    else:
        header = "Answer the question as truthfully as possible using the provided context. You should use as much detail from the given context as possible when answering the question."
        header += "If the answer is not contained within the text below, say 'I don't know.' followed by the all the text in the 'Context' section, with their respective URLs (preceeded by 'Here is the closest information I could find to your question\\n\\n:'). "
        header += "Within the context are URLs. If an answer if found within a relevant section, return the answer and then three line breaks and then the text 'More info:' followed by the URL."
    
    header += ""

    header += "\n\nContext:\n"    
    header += "".join(chosen_sections) + "\n\n"
    header += "Q: " + question

    return header
     

def answer_question_with_context(
    question: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False,
    print_question: bool = True,
    return_answer: bool = False,
    imagine = False,
) -> str:
    answer = ""
    prompt = construct_prompt(
        question=question,
        context_embeddings=document_embeddings,
        df=df,
        imagine=imagine
    )
    
    if show_prompt:
        if return_answer:
            answer += f"{prompt}\n\n"
        else:
            print(f"\n\n{prompt}")
        
    if(print_question):
        print(f"\n\nQuestion: {question}")


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
        print(f"Answer: {answer}")

contentDir = args.dir.rstrip("/")
contentsFile = f"{contentDir}/contents.csv"
embeddingsFile = f"{contentDir}/embeddings.csv"

if not os.path.exists(contentsFile) or not os.path.exists(embeddingsFile):
    print("Error: Both contents.csv and embeddings.csv must exist in the provided directory")
    sys.exit()

start_time = time.time()
# Fetch the embeddings from the CSV
document_embeddings = load_embeddings(embeddingsFile)
load_time = time.time() - start_time
print(f"Embeddings loaded in {round(load_time,2)} seconds")

df = pd.read_csv(contentsFile)
df = df.set_index(["title", "heading"])

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
        data = payload['data']

        # Get the user's information from the Slack API
        user_info = client.users_info(user=data['user'])
        user_question = data['text']
        thread_ts = data.get("ts")

        # Check if the bot was mentioned and ensure it is not the bot talking to itself!
        if 'bot_id' not in data and bot_id in data.get('text', ''):
            # Extract the username from the returned JSON object
            username = user_info['user']['name']
            print("------")
            print(f"User {username} asks \"{user_question}\"")
            channel_id = data['channel']
            # Allow some hallucinations if the CLI or user wants it
            imagine = args.imagine or '[--imagine]' in data.get('text', '')
            show_prompt = args.show_prompt or '[--show_prompt]' in data.get('text', '')
            
            holding_message = "Let me look that up for you! This might take a second..."
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
                document_embeddings=document_embeddings,
                show_prompt=show_prompt,
                print_question=False,
                return_answer=True,
                imagine=imagine
            )

            # Return the answer back to the Slack channel
            client.chat_postMessage(
                channel=channel_id,
                text=answer,
                as_user=True,
                thread_ts=thread_ts
            )
            execution_time = (time.time() - start_time) * 1000
            print(f"Responded in {round(execution_time)}ms")

    # Start the slack client
    rtm_client = slack.RTMClient(token = slack_token)
    rtm_client.start()
else:
    print('Answering question...')
    answer_question_with_context(
        question=args.question,
        df=df,
        document_embeddings=document_embeddings,
        show_prompt=args.show_prompt,
        print_question=True,
        return_answer=False,
        imagine=args.imagine
    )
    exit()