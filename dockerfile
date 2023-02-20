#see https://github.com/ht2/gpt_content_indexing
# to use this docler file fits build an image. Before doing that consider delete any older images
# docker rmi -f $(docker images -aq)  
# docker image build -t gtp_experiment .  

#Deriving the latest base image
FROM python:latest

#Labels as key value pair
LABEL Maintainer="jeremy.mindtools"

# Any working directory can be chosen as per choice like '/' or '/home' etc
WORKDIR /usr/app/src

#to COPY the remote file at working directory in container
COPY *.py ./
RUN mkdir -p ./output/default

#install python dependencies spcified in requirments
ADD requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# get from https://platform.openai.com/account/api-keys
ENV OPENAI_API_KEY=<your openAI API key goes here>

# get api key here https://id.atlassian.com/manage-profile/security/api-tokens
ENV CONFLUENCE_USERNAME=<your atlassian login/email goes here>
ENV CONFLUENCE_API_KEY=<your atlassian API key goes here>

# use whatever values you want
ENV CONFLUENCE_URL=https://emeraldworks.atlassian.net
ENV CONFLUENCE_SPACE=buildChapter

# missing dependency for python natural language thing
RUN python -m nltk.downloader punkt

#create and index the content to be ingested by the model
RUN python ./index_content.py --spaces=$CONFLUENCE_SPACE --max_pages=20 --out ./output/default/contents.csv
RUN python ./create_embeddings.py --file ./output/default/contents.csv --out ./output/default/embeddings.csv

#output env vars
RUN python ./test_api_keys.py

# to run the scripts within use following at terminal:
# docker run gtp_experiment python ask_question.py --question 'Why did we start Nova?'