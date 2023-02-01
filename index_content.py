import os
import pandas as pd
import csv
import html2text
import sys
import requests
from atlassian import Confluence
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextLineHorizontal, LTTextBoxHorizontal, LTChar

from io import StringIO
from pprint import pprint
from bs4 import BeautifulSoup
import argparse
from transformers import GPT2TokenizerFast
from typing import Tuple
from nltk.tokenize import sent_tokenize

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument with a flag and a name
parser.add_argument("--spaces", nargs="*", default=["STRM"], help="Specify the Confluence Space you want to index")
parser.add_argument("--zendesk", nargs="*", default=["learningpool"], help="Specify the Zendesk domains you want to index")
parser.add_argument("--max_pages", default=1000, help="The maximum amount of Space pages to index")
parser.add_argument("--out", default="./output/default/contents.csv", help="Specify the filename to save the content")
parser.add_argument("--min_tokens", default=20, help="Remove content with less than this number of tokens")
parser.add_argument("--input", default="./input", help="Folder to ingest CSVs from. Rows should be in the format 'heading,answers,answers,...'")
parser.add_argument("--use_dirs", default=False, help="Use the folder structure (./product/area.csv)")
parser.add_argument("--pdf_content_fontsize", default=12, help="Content greater than this fontsize will be considered as a header")

args = parser.parse_args()
max_pages = int(args.max_pages)
pdf_content_fontsize = int(args.pdf_content_fontsize)

# Connect to Confluence
confluence = Confluence(url='https://learninglocker.atlassian.net', username=os.environ.get('CONFLUENCE_USERNAME'), password=os.environ.get('CONFLUENCE_API_KEY'))

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."

    return long_text


def extract_html_content(
  title_prefix: str,
  page_title: str,
  html: str,
  url: str
):
  ntitles, nheadings, ncontents, nurls = [], [], [], []

  soup = BeautifulSoup(html, 'html.parser')
  headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

  prev_heading = []

  # Iterate through all headings and subheadings
  for h in headings:
    # Extract the heading text and remove HTML
    heading = html2text.html2text(str(h)).strip()

    # Initialize the content list
    content = []

    # Find the next heading or subheading
    next_h = h.find_next(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    actual_heading = heading.lstrip('#').lstrip(' ')

    # Iterate through all siblings until the next heading or subheading is reached
    for sibling in h.next_siblings:
      if sibling == next_h:
        break

      # If the sibling is a tag, extract the text and remove HTML
      if sibling.name:
        para = html2text.html2text(str(sibling)).strip()
        if len(para) > 0:
          content.append(para)

    # If there are content entries, join them all together, clean up for utf-8 and write the row
    if len(content) > 0:
      content = "".join(content).replace("\n", "").encode('utf-8').decode('utf-8')

      # If there are headings above this one without content, we concat them here
      if len(prev_heading) > 0:
        full_heading = " - ".join(prev_heading) + " - " + actual_heading
      else:
        full_heading = actual_heading

      title = f"{title_prefix} - {page_title}"
      # Store the extracted title, heading, content
      ntitles.append(title)
      nheadings.append(full_heading)
      ncontents.append(f"{title} - {full_heading} - {content}")
      nurls.append(url)
      prev_heading = []
    else:
      # Otherwise, we store this heading to append to the next sibling with content
      prev_heading.append(actual_heading)
  
  # Return the 3 arrays of titles, headings and content
  return (ntitles, nheadings, ncontents, nurls)

def count_content_tokens(
  ntitles: list,
  nheadings:list,
  ncontents: list,
  nurls: list
):
  # count the tokens of each section
  ncontent_ntokens = [
      count_tokens(c) # Add the tokens from the content
      + 4
      + count_tokens(" ".join(t.split(" ")[1:-1])) # Add the tokens from the titles
      + count_tokens(" ".join(h.split(" ")[1:-1])) # Add the tokens from the headings
      + count_tokens(" ".join(u.split(" ")[1:-1])) # Add the tokens from the url
      - (1 if len(c) == 0 else 0)
      for t, h, c, u in zip(ntitles, nheadings, nurls, ncontents)
  ]
  # Create a tuple of (title, section_name, content, number of tokens)
  outputs = []
  outputs += [(t, h, u, c, tk) if tk<max_len 
              else (h, reduce_long(c, max_len), count_tokens(reduce_long(c,max_len))) 
                  for t, h, u, c, tk in zip(ntitles, nheadings, nurls, ncontents, ncontent_ntokens)]
  return outputs


def extract_sections(
  space: str,
  limit: int = max_pages
):
  ntitles, nheadings, ncontents, nurls = [], [], [], []

  confluence_space = confluence.get_space(space_key=space)
  space_title = confluence_space['name']

  print(f"Fetching up to {limit} pages from '{space_title}'...")

  # Search for all pages in a given space
  results = confluence.get_all_pages_from_space(space=space, start=0, limit=limit)

  page_ids = []
  for result in results:
      page_ids.append(result["id"])

  # Iterate through the list of Confluence pages
  for page_id in page_ids:
      # Fetch the Confluence page
      page = confluence.get_page_by_id(page_id=page_id, expand="body.storage")

      # Extract the page title and content
      page_title = page['title']
      page_html = page['body']['storage']['value']
      page_url = page['_links']['base'] + page['_links']['webui'];
      
      pageTitles, pageHeadings, pageContent, pageUrls = extract_html_content(space_title, page_title, page_html, page_url)
      ntitles += pageTitles
      nheadings += pageHeadings
      ncontents += pageContent
      nurls += pageUrls

  return count_content_tokens(ntitles, nheadings, ncontents, nurls) 


def extract_zendesk_domain(
  zendesk_domain: str,
  limit: int = max_pages
):
  ntitles, nheadings, ncontents, nurls = [], [], [], []

  total_pages = 0;
  URL = f"https://{zendesk_domain}.zendesk.com/api/v2/help_center/en-us"
  
  print(f"Fetching up to {limit} pages from 'https://{zendesk_domain}.zendesk.com'...")

  # Fetch the Categories from Zendesk
  cat_response = requests.get(URL + '/categories.json')
  cat_data = cat_response.json()
  for category in cat_data['categories']:
    category_title = category['name']

    # Fetch the sections within the categories
    sections_response = requests.get(URL + '/categories/' + str(category['id']) + '/sections.json')
    sections_data = sections_response.json()
    for section in sections_data['sections']:
      page_title = section['name']
      
      # Fetch the articles within the section
      articles_response = requests.get(URL + '/sections/' + str(section['id']) + '/articles.json')
      articles_data = articles_response.json()

      for article in articles_data["articles"]:
        page_title += " - " + article['title']
        page_html = article['body']
        page_url = article['html_url']

        if (page_html is not None and total_pages < limit ):
          pageTitles, pageHeadings, pageContent, pageUrls = extract_html_content(category_title, page_title, page_html, page_url)
          ntitles += pageTitles
          nheadings += pageHeadings
          ncontents += pageContent
          nurls += pageUrls
          total_pages += 1
      
      if (articles_data['next_page'] is not None):
        pprint('TODO! But have not seen multiple pages yet at this level (due to using sections...)')
  
  return count_content_tokens(ntitles, nheadings, ncontents, nurls)

def extract_csvfile(subdir, file):
    ntitles, nheadings, ncontents, nurls = [], [], [], []
    csv_filepath = os.path.join(subdir, file)
    print(f"Loading data from {csv_filepath}")
    subdir_name = os.path.basename(subdir)
    file_name = os.path.splitext(file)[0]

    if args.use_dirs == False:
      product = input(f"Please enter the product NAME for this file (default: {subdir_name}): ")
      if not product:
        product = subdir_name
      product_area = input(f"Please enter the product AREA for this file (default: {file_name}): ")
      if not product_area:
        product_area = file_name
    else:
      product = subdir_name
      product_area = file_name
    
    title = f"{product} - {product_area}"

    with open(csv_filepath, 'r', encoding='utf-8') as csv_file:
      csv_reader = csv.reader(csv_file)
      for row in csv_reader:
        heading = row[0]
        ntitles.append(title)
        nheadings.append(heading)
        content = f"{title} - {heading} - {row[1]}"
        for i in range(2, len(row)):
          if row[i]:
            content += ' ' + row[i]
        ncontents.append(content)
        nurls.append(file)
    return count_content_tokens(ntitles, nheadings, ncontents, nurls)


import PyPDF2

def index_pdf_content(subdir, file):
    filepath = os.path.join(subdir, file)
    ntitles, nheadings, ncontents, nurls = [], [], [], []    
    print(f"Loading data from {filepath}")
    subdir_name = os.path.basename(subdir)
    file_name = os.path.splitext(file)[0]
    if args.use_dirs == False:
      product = input(f"Please enter the product NAME for this file (default: {subdir_name}): ")
      if not product:
        product = subdir_name
      product_area = input(f"Please enter the product AREA for this file (default: {file_name}): ")
      if not product_area:
        product_area = file_name
    else:
      product = subdir_name
      product_area = file_name

    title = f"{product} - {product_area}"

    # open the pdf file
    with open(filepath, 'rb') as pdf_file:
        laparams = LAParams()
        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        pages = PDFPage.get_pages(pdf_file)

        content = {}
        current_headings = []
        prev_heading_size = 0
        for page in pages:
            interpreter.process_page(page)
            layout = device.get_result()
            for element in layout:
                if isinstance(element, LTTextBoxHorizontal):
                    for text_line in element:
                        if isinstance(text_line, LTTextLineHorizontal):
                            is_heading = False
                            for char in text_line:
                                if isinstance(char, LTChar):
                                    fontsize = char.matrix[3]
                                    if fontsize > pdf_content_fontsize:
                                        is_heading = True
                                        break
                            if is_heading:
                                heading = text_line.get_text().replace('\n', '')
                                if fontsize == prev_heading_size and len(current_headings) > 0:
                                    current_headings.pop()
                                elif fontsize > prev_heading_size:
                                    current_headings = []
                                current_headings.append(heading)
                                prev_heading_size = fontsize
                                break
                        
                    line_text = element.get_text().replace('\n', '')
                    key = ' - '.join(current_headings)
                    if key not in content:
                        content[key] = [line_text]
                    else:
                        content[key].append(line_text)

        for heading in content:
            # pprint(f"adding {heading}")
            ntitles.append(title)
            nheadings.append(heading)
            content_text = " ".join(content[heading])
            ncontents.append(f"{heading} - {content_text}")
            nurls.append(f"{file_name} - {heading}")

    return count_content_tokens(ntitles, nheadings, ncontents, nurls)

# Define the maximum number of tokens we allow per row
max_len = 1500

# For each Space, fetch the content and add to a list(title, heading, content, tokens)
res = []

for space in args.spaces:
  print(f"INDEXING CONTENT FROM CONFLUENCE: {space}")
  res += extract_sections(space)

for domain in args.zendesk:
  print(f"INDEXING CONTENT FROM ZENDESK: {domain}.zendesk.com")
  res += extract_zendesk_domain(domain)

if os.path.isdir(args.input):
  for subdir, dirs, files in os.walk(args.input):
    for file in files:
      if file.endswith(".csv"):
        res += extract_csvfile(subdir, file)
      elif file.endswith(".pdf"):
        res += index_pdf_content(subdir, file)


  
# Remove rows with less than 40 tokens
df = pd.DataFrame(res, columns=["title", "heading", "url", "content", "tokens"])
df = df[df.tokens > args.min_tokens]
df = df.drop_duplicates(['title','heading'])
df = df.reset_index().drop('index',axis=1) # reset index
print(df.head())

# Store the content to a CSV
df.to_csv(args.out, index=False)
print(f"Done! File saved to {args.out}")