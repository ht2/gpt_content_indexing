import os
import csv
import html2text
import sys
from atlassian import Confluence
import openai
from pprint import pprint
from bs4 import BeautifulSoup


# Replace YOUR_API_TOKEN, YOUR_CONFLUENCE_URL, and YOUR_OPENAI_API_KEY with your own values
confluence = Confluence(url='https://learninglocker.atlassian.net', username=os.environ.get('CONFLUENCE_USERNAME'), password=os.environ.get('CONFLUENCE_API_KEY'))

# Search for all pages in a given space
results = confluence.get_all_pages_from_space(space="STRM", start=0, limit=1000)

page_ids = []
for result in results:
    page_ids.append(result["id"])

# Create a CSV file to store the extracted content
with open('output/confluence_content.csv', 'w', encoding="utf-8", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['title', 'heading', 'content'])
    writer.writeheader()

    # Iterate through the list of Confluence pages
    for page_id in page_ids:
        # Fetch the Confluence page
        page = confluence.get_page_by_id(page_id=page_id, expand="body.storage")

        # Extract the page title and content
        page_title = page['title']
        soup = BeautifulSoup(page['body']['storage']['value'], 'html.parser')
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

            # Write the extracted title, heading, content
            writer.writerow({'title': page_title, 'heading': full_heading, 'content': content})
            prev_heading = []
          else:
            # Otherwise, we store this heading to append to the next sibling with content
            prev_heading.append(actual_heading)

print('Done!')