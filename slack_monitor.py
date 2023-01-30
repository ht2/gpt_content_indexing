import os
from pprint import pprint
import slack
import subprocess

# Initialize the Slack API client
client = slack.WebClient(os.environ.get('SLACK_BOT_API_KEY'))
bot_id = os.environ.get('SLACK_BOT_ID')


# Listen for messages in the "qa-lphandbook" channel
@slack.RTMClient.run_on(event='message')
def respond_to_message(**payload):
    data = payload['data']
    user_question = data['text']

    if 'bot_id' not in data and bot_id in data.get('text', ''):
        pprint(f"Answering {user_question}")
        channel_id = data['channel']

        # Return the answer back to the Slack channel
        client.chat_postMessage(
            channel=channel_id,
            text=f"Let me look that up for you! Give me a second...",
            as_user=True
        )
        # Call the external Python script
        result = subprocess.run(["./venv/Scripts/python", "ask_question.py", "--no-stream", "--answer_only", "--file", "./output/pdftest.csv", "--embeddings", "./output/pdfembeddings.csv", "--question", user_question], capture_output=True, text=True)
        pprint(result)
        # Check the return code to see if the script ran successfully
        if result.returncode == 0:
            # Return the output of the script
            response = result.stdout
        else:
            # Return an error message
            response = "Error: The script failed with return code {}".format(result.returncode)

        # Return the answer back to the Slack channel
        client.chat_postMessage(
            channel=channel_id,
            text=response,
            as_user=True
        )

slack_token = os.environ.get('SLACK_BOT_API_KEY')
rtm_client = slack.RTMClient(token = slack_token)
rtm_client.start()