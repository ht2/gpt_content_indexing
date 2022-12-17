# Confluence Space indexing & embeddings for Q&A through GPT3

The aim of this project is to fetch all the content from a given Confluence space and index the content found within the subheadings in each page of the Space. 

The content is output to CSV where it can then be parsed through the OpenAI embeddings model to provide vector scores across all of the contexts.

Once we have an embeddings database of the content, we can then use an embeddings compare search to retrieve the best context for a given question, and use this as the context when submitting a question to the GPT-3 model.

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
- [ ] Create script to take input question from user, search indexed content, select most similar content and apply it as context back to GPT3 (davinci?) alongside original question to attempt to answer from knowledge base

#### Stretch goals

- Index multiple spaces into single embeddings database
- Index other knowledge sources such as Slack support channels
- Fine tune model rather than just finding and providing context (although this is still a recommended approach)
- Automate Q&A through Slackbot


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

## Example

Execute [extract_confluence.py](extract_confluence.py) to generate a CSV of content from the space in `output/`

```bash
python index_space.py --space="STRM" --max_pages=1000
```

`--space` is required

`--max_pages` is not required; defaults to 1000


## References

A lot of the work to date has been inspired by the notebooks and cookbooks provided by OpenAI:

- [Question Answering using Embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)
- [Fine tune Q&A examples](https://github.com/openai/openai-cookbook/tree/main/examples/fine-tuned_qa)