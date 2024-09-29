import json
import urllib.request
from io import BytesIO
from PyPDF2 import PdfReader

def extract_text_pypdf2(pdf_path):
    response = urllib.request.urlopen(pdf_path)
    pdf_data = response.read()
    pdf_bytes = BytesIO(pdf_data)
    reader = PdfReader(pdf_bytes)
    pageObj = reader.pages[0]
    print(pdf_path, pageObj.extract_text())  # Extract text from the first page

    text = ""
    # Iterate through pages
    # for page_num in range(len(reader.pages)):
    #     page = reader.pages[page_num]
    #     text += page.extract_text()
    # return text

# Example usage
# pdf_path = 'example.pdf'
# text = extract_text_pypdf2(pdf_path)
# print(text)

with open('data.json') as f:
    items = json.load(f)
    for item in items:
        links_arr = item['info']
        for link in links_arr:
            extract_text_pypdf2(link)