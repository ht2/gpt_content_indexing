# Confluence Space indexing & embeddings for Q&A through GPT3

**The aim of this project is to provide a naturally queirable knowledge base using an indexed set of content from Confluence alongside GPT3 models.**


We will accomplish this by fetching all the content from a given Confluence space and indexing the content found within the subheadings in each page of the Space into a CSV. The CSV can then be parsed through the OpenAI embeddings model to provide vector scores across all of the contexts.

Once we have an embeddings database of the content, we can then use an embeddings compare search to retrieve the best context for a given question, and use this as the context when submitting a question to the GPT-3 model. Further work can then be done to allow the script and indexed content to be accessible through interfaces such as Slack.

## Example use case

Imagine a user posing the question:

> Question: Can I still see my completion if I delete my enrolment?

Using the OpenAI embeddings API, we would return the "embedding" for this question and then compare it against the embeddings generated from our knowledge bank (an indexed Confluence Space, e.g. STRM).

In this case, we would find the following answer [from our FAQs](https://learninglocker.atlassian.net/wiki/spaces/STRM/pages/1014333441/FAQs#What-happens-to-completions-if-you-expire-or-delete-an-enrolment%3F)

![What happens to completions if you expire or delete an enrolment? If you expire all enrolments for a user, then their completions are retained in the LX leaderboards but the user does not count against active users licences. If you delete all enrolments for a user, then the completions ARE NOT shown on the leaderboard and of course, they do not count as an active user. But if you re-enrol the user, then their completion is resurfaced too.](https://user-images.githubusercontent.com/1352590/208269780-283539a3-31da-419a-8210-57fee625dec5.png)

We would then use this context to generate a better context when querying the GPT3 model:

![image](https://user-images.githubusercontent.com/1352590/208269756-da67a4f7-5b7b-4dcb-9f22-d1df4e591f26.png)


## Runtime example

Execute [index_space.py](index_space.py) to generate a CSV of content from the space in `output/`

```bash
python index_space.py --space="STRM" --max_pages=1000
```

`--space` is required

`--max_pages` is not required; defaults to 1000

## Requirements

- Python 3.x
- Check [requirments.txt](requirements.txt) for pip modules

### OpenAI

You will require an OpenAI account and your OpenAI API key included in a `OPENAI_API_KEY` environment variable.

### Atlassion/Confluence

The script uses the Atlassian API to fetch pages from Confluence. Ensure the following ENV variables are included in your system:

`CONFLUENCE_USERNAME=your confluence username`

`CONFLUENCE_API_KEY=your confluence api key`

Instructions on generating a Confluence API token: https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/


## Todo

- [x] Index a Confluence Space
  - Parse through all pages within a space
  - Return out all content, stripped of HTML and indexed by heading/subheading
- [ ] Count tokens in each indexed row of content
  - Use `tokenizer` from GPT2TokenizerFast ([tokenizers](https://github.com/huggingface/tokenizers))
  - Discard content where too small (<40?)
  - See https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-1-collect-data.ipynb for example
- [ ] Parse indexed content through embeddings API to generate vectors
  - https://openai.com/blog/new-and-improved-embedding-model/
  - https://beta.openai.com/docs/guides/embeddings
- [ ] Create script to take input question from user and return a useful answer!
  - Search indexed content using question's embedding and select most similar content
  - Apply it as context back to GPT3 (davinci?) alongside original question to attempt to answer from knowledge base
  -

#### Stretch goals

- Index multiple spaces into single embeddings database
- Index other knowledge sources such as Slack support channels
- Fine tune model rather than just finding and providing context (although this is still a recommended approach)
- Automate Q&A through Slackbot



## References

A lot of the work to date has been inspired by the notebooks and cookbooks provided by OpenAI:

- [Question Answering using Embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)
- [Fine tune Q&A examples](https://github.com/openai/openai-cookbook/tree/main/examples/fine-tuned_qa)
