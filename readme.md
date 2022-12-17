# Confluence Embeddings

The aim of this project is to fetch all the content from a given Confluence space and index the content found within the subheadings in each page of the Space. 

The content is output to CSV where it can then be parsed through the OpenAI embeddings model to provide vector scores across all of the contexts.

Once we have an embeddings database of the content, we can then use an embeddings compare search to retrieve the best context for a given question, and use this as the context when submitting a question to the GPT-3 model.