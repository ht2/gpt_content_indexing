# Content indexing & embeddings for Q&A through GPT3

Welcome to our open source repo that provides an interface to query knowledge bases using an indexed set of content from various sources such as files, Confluence, and Zendesk using GPT3/4 models.

Our system fetches all the content from a given directory, Confluence spaces, and Zendesk domain and indexes the content found within the subheadings in each page of the Space/Domain into a CSV. The CSV is then parsed through the OpenAI embeddings model to provide vector scores across all of the contexts.

With our embeddings database of the content, we can use an embeddings compare search to retrieve the best context for a given question and use this context when submitting a question to OpenAI models.

## Example Use Case

Imagine a user posing the question:

> Question: Can I still see my completion if I delete my enrolment?

Using the OpenAI embeddings API, we would return the "embedding" for this question and then compare it against the embeddings generated from our knowledge bank (an indexed Confluence Space, e.g. STRM).

In this case, we would find the following answer [from our FAQs](https://learninglocker.atlassian.net/wiki/spaces/STRM/pages/1014333441/FAQs#What-happens-to-completions-if-you-expire-or-delete-an-enrolment%3F)

![What happens to completions if you expire or delete an enrolment? If you expire all enrolments for a user, then their completions are retained in the LX leaderboards but the user does not count against active users licences. If you delete all enrolments for a user, then the completions ARE NOT shown on the leaderboard and of course, they do not count as an active user. But if you re-enrol the user, then their completion is resurfaced too.](https://user-images.githubusercontent.com/1352590/208269780-283539a3-31da-419a-8210-57fee625dec5.png)

We would then use this context to generate a better context when querying the GPT3 model:

![image](https://user-images.githubusercontent.com/1352590/208269756-da67a4f7-5b7b-4dcb-9f22-d1df4e591f26.png)

### Example running in code

The same question posed through this repo:

```bash
$ python ask_question.py --allow_hallucinations True --question "Can I still see my completion if I delete my enrolment?"

Question: Can I still see my completion if I delete my enrolment?

Answer: No, if you delete all enrolments for a user, then the completions ARE NOT shown on the leaderboard. More info: https://learninglocker.atlassian.net/wiki/spaces/STRM/pages/1014333441/FAQs
```

![image](https://user-images.githubusercontent.com/1352590/210773311-16b3a41d-11dc-48a6-9530-5ea0f2306f75.png)

---
## index_content.py - Generate CSV of Spaces, Zendesk, and Custom CSV Content

Use this script to generate a CSV file of Confluence Spaces, Zendesk content, and custom CSV content. By default, the script saves the output to `./output/default/contents.csv`. You can specify a different output file using the `--out` flag.

#### Example:

```bash
python index_content.py --spaces=STRM LL --max_pages=10 --zendesk learningpool --out ./output/default/contents.csv
```

#### Arguments

- `--spaces`: A space-separated list of Confluence Spaces to index. You can specify multiple spaces by separating them with a space. Defaults to `["STRM"]`.
- `--zendesk`: A list of Zendesk domains to index. You can specify multiple domains by separating them with a space. Defaults to `["learningpool"]`.
- `--max_pages`: The maximum number of pages to index per Confluence Space. Defaults to `1000`.
- `--out`: The name of the output file. Defaults to `./output/default/contents.csv`.
- `--min_tokens`: The minimum number of tokens that a given bit of content must have before it is included in the output file. Defaults to `20` (approximately 60 characters).

#### CSV Import

For CSVs included in the `csv_input` directory, they will be iterated over and imported.

Content can now be imported from files using the following options:

- `--input`: The folder to ingest CSVs from. Defaults to `./input`. Rows should be in the format `heading,answers,answers,...`.
- `--use_csv_dirs`: If `True`, use the folder structure (`./input/product/area.csv`) to prefill the title (e.g. "product") and subtitle (e.g. "area") for imported CSVs. Otherwise, prompts for each file found. Defaults to `False`.

#### Example:

```bash
python index_content.py --input ./input --use_csv_dirs
```

**Note:** If you specify both `--input` and `--spaces`/`--zendesk` arguments, the script will include content from both sources in the output file. Use the `--use_csv_dirs` flag if you want to use the folder structure to group content.


---

## create_embeddings.py - Generate Embeddings from Output File

Use this script to generate embeddings from an input CSV file. By default, the script outputs embeddings to a file in `./default/output/embeddings.csv`. You can specify a different output file using the `--out` flag.

#### Example:

```bash
python create_embeddings.py --file ./output/default/contents.csv --out ./output/default/embeddings.csv
```

#### Arguments

- `--file`: The path to the input CSV file. Defaults to `./output/default/contents.csv`.
- `--embedding_type`: The format to save embeddings in. You can choose between `csv` and `pinecone`. Defaults to `csv`.
- `--out`: The name of the output file. Defaults to `./output/default/embeddings.csv`.
- `--pinecone_mode`: The mode to upsert or replace embeddings in a Pinecone index. You can choose between `upsert` and `replace`. Defaults to `replace`.
- `--pinecone_index`: The name of the Pinecone index to use. Defaults to `default`.
- `--pinecone_namespace`: The name of the Pinecone namespace to use. Defaults to `content`.

**Note:** If you choose `pinecone` as the `--embedding_type`, you must specify `--pinecone_index` and `--pinecone_namespace` arguments. 

#### Example:

```bash
python create_embeddings.py --file ./output/default/contents.csv --embedding_type pinecone --pinecone_index my_index --pinecone_namespace my_namespace
```


---

## ask_question.py - Ask a question

You can use this script to get an answer to your question. By default, it looks for a directory named `./output/default/` and loads `contents.csv` and `embeddings.csv` from there. You can specify a different directory using the `--dir` flag.

```bash
python ask_question.py --question "How much does an elephant weigh?"
```

#### Arguments

- `--question`: Specify the question you want to ask.
- `--dir`: Specify the directory containing the `contents.csv` and `embeddings.csv` files. Defaults to `./output/default/`.
- `--debug`: Enable debug mode (show prompts and other info).
- `--imagine`: Don't restrict answers to be based from the provided context.
- `--show_prompt`: Show the full prompt used to generate the answer.
- `--stream`: Stream the output from GPT directly to the terminal in real-time.
- `--experiment_hyde`: Generate an answer from the question and use that for embedding lookup. This mode is based on Hypothetical Document Embeddings and generates a hypothetical answer to the original question, which is then used to search through the contents. Credit to https://arxiv.org/pdf/2212.10496.pdf via https://twitter.com/mathemagic1an/status/1615378778863157248.
- `--custom_instructions`: Inject a custom set of instructions before the context.

#### Embedding Options

- `--embedding_type`: Format to save embeddings in. You can choose between `csv` and `pinecone`.
- `--pinecone_index`: The Pinecone index to use.
- `--pinecone_namespace`: The Pinecone namespace to use.
- `--pinecone_top_k`: The number of results to return from the Pinecone index.

#### Completion Options

- `--completion_type`: The type of completion to use. You can choose between `text` and `chat`. For best results, use chat with the `gpt-4` model.
- `--text_model`: The text completions model to use. Defaults to `text-davinci-003`.
- `--chat_model`: The chat completions model to use. Defaults to `gpt-4`.
- `--max_tokens`: The maximum number of tokens to generate. Defaults to `600`.
- `--max_context`: The maximum length of content to include. Defaults to `1000`.

---

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

---

## Todo

- [x] Index a Confluence Space
  - Parse through all pages within a space
  - Return out all content, stripped of HTML and indexed by heading/subheading
- [x] Count tokens in each indexed row of content
  - Use `tokenizer` from GPT2TokenizerFast ([tokenizers](https://github.com/huggingface/tokenizers))
  - Discard content where too small (<40?)
  - See https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-1-collect-data.ipynb for example
- [x] Parse indexed content through embeddings API to generate vectors
  - https://openai.com/blog/new-and-improved-embedding-model/
  - https://beta.openai.com/docs/guides/embeddings
- [x] Create script to take input question from user and return a useful answer!
  - Search indexed content using question's embedding and select most similar content
  - Apply it as context back to GPT3 (davinci?) alongside original question to attempt to answer from knowledge base
- [x] Utilise a Vector Database
- [x] Implement Chat Completions API (GPT-4 compatibility)

#### Stretch goals

- [x] Index multiple spaces into single embeddings database
- [x] Index other knowledge sources
  - [x] Academy: https://learningpool.zendesk.com/api/v2/help_center/en-us/articles.csv
  - [x] CSV
  - [x] PDF
  - [x] TXT

---

## Known issues and workarounds

### Can't find Rust compiler
The transformers module, used to count tokens of content, requires rust to compile.

Solution: Install rust


### Resource punkt not found
See reference issue here: https://github.com/delip/PyTorchNLPBook/issues/14

Solution: Execute this in the Python terminal:
```python
import nltk
nltk.download('punkt')
```


## References

A lot of the work to date has been inspired by the notebooks and cookbooks provided by OpenAI:

- [Question Answering using Embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)
- [Fine tune Q&A examples](https://github.com/openai/openai-cookbook/tree/main/examples/fine-tuned_qa)
