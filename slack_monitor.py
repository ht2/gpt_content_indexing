import argparse
import os
import sys
from pprint import pprint
import slack
import subprocess

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Initialize the Slack API client
client = slack.WebClient(os.environ.get('SLACK_BOT_API_KEY'))
bot_id = os.environ.get('SLACK_BOT_ID')

parser.add_argument("--dir", help="Specify the directory containing the contents.csv and embeddings.csv")
args = parser.parse_args()

contentDir = args.dir.rstrip("/")
contentsFile = f"{args.dir}/contents.csv"
embeddingsFile = f"{args.dir}/embeddings.csv"

if not os.path.exists(contentsFile) or not os.path.exists(embeddingsFile):
    print("Error: Both contents.csv and embeddings.csv must exist in the provided directory")
    sys.exit()


pprint("LISTENING FOR MESSAGES...")
# Listen for messages in the "qa-lphandbook" channel
@slack.RTMClient.run_on(event='message')
def respond_to_message(**payload):
    data = payload['data']

    # Get the user's information from the Slack API
    user_info = client.users_info(user=data['user'])
    user_question = data['text']
    thread_ts = data.get("ts")

    if 'bot_id' not in data and bot_id in data.get('text', ''):
        # Extract the username from the returned JSON object
        username = user_info['user']['name']
        pprint(f"User {username} asks \"{user_question}\"")
        channel_id = data['channel']

        # Return the answer back to the Slack channel
        client.chat_postMessage(
            channel=channel_id,
            text=f"Let me look that up for you! Give me a second...",
            as_user=True,
            thread_ts=thread_ts
        )
        # Call the external Python script
        result = subprocess.run(["./venv/Scripts/python", "ask_question.py", "--no-stream", "--answer_only", "--dir", contentDir, "--question", user_question], capture_output=True, text=True)
        # Check the return code to see if the script ran successfully
        if result.returncode == 0:
            # Return the output of the script
            response = result.stdout
        else:
            # Return an error message
            pprint(result)
            response = "Oops. James seems to have broken something - go easy on him"

        # Return the answer back to the Slack channel
        client.chat_postMessage(
            channel=channel_id,
            text=response,
            as_user=True,
            thread_ts=thread_ts
        )

slack_token = os.environ.get('SLACK_BOT_API_KEY')
rtm_client = slack.RTMClient(token = slack_token)
rtm_client.start()