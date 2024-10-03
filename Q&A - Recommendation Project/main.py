import urllib.request
from io import BytesIO
from PyPDF2 import PdfReader
import ssl
import certifi
import json
import random

from transformers import LlamaTokenizer, LlamaForCausalLM
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

context = ssl.create_default_context(cafile=certifi.where())
pc = Pinecone(api_key='5f00d642-14fd-4fd3-acb4-60f9976000ea')
# Create Pinecone index (only once)
index_name = "deal-index"
indexInstance = ''
extracted_data_collection = []
processed_data = []

def find_or_create_pinecode_index():
    if index_name not in pc.list_indexes().names():
      pc.create_index(name=index_name,
                      dimension=384,  # Replace with your model dimensions
                      metric="cosine",  # Replace with your model metric
                      spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                      ))

# index = pc.Index(index_name)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = LlamaTokenizer.from_pretrained("/Users/mpichikala/personal/llama-2/")
model = LlamaForCausalLM.from_pretrained("/Users/mpichikala/personal/llama-2/")

def ask_llama(question, context):
  # Prepare input for Llama
  input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
  inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

  # Generate answer
  outputs = model.generate(inputs['input_ids'], max_length=200)
  return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_text_pypdf2(deal, content, pdf_path):
  response = urllib.request.urlopen(pdf_path, context=context)
  pdf_data = response.read()
  pdf_bytes = BytesIO(pdf_data)
  reader = PdfReader(pdf_bytes)
  text = ""
  # Iterate through pages
  for page_num in range(len(reader.pages)):
      page = reader.pages[page_num]
      text += page.extract_text()
  return { 'deal': deal, 'content': content, 'text': text }

def store_embedding_in_pinecone(deal_id, doc_id, embedding):
  print(f"inserting {deal_id} {doc_id} {embedding}")
  pc.Index(index_name).upsert(vectors=[{ 'id': str(random.randint(1, 10)), 'metadata': {'deal': deal_id, 'content': doc_id,  }, 'values': embedding }])

def get_embeddings(text):
  return embedding_model.encode(text)

def query_pinecone(deal_id, question_embedding):
  # Search for the top 3 most relevant documents in the deal
  results = pc.Index(index_name).query(vector=question_embedding.astype(float).tolist(), include_values=True, include_metadata=True, top_k=10, filter={"deal": {"$eq": deal_id}})
  return results

def process_files():
  with open('data.json') as f:
    items = json.load(f)
    for item in items:
      links_arr = item['contents']
      for link in links_arr:
        extracted_object = extract_text_pypdf2(item['deal_id'], link['id'], link['url'])
        extracted_data_collection.append(extracted_object)
    return extracted_data_collection

def load_and_get_embeddings(data):
  extracted_data_collection = data
  for item in extracted_data_collection:
    store_embedding_in_pinecone(item['deal'], item['content'], get_embeddings(item['text']))

def extract_and_load_info_file():
  # loading extracted data to json file
  with open('extracted_data.json') as f:
    items = json.load(f)
    if len(items) > 0:
      processed_data = items
      print("extracted_data.json is not empty")
    else:
      processed_data = process_files()
      with open('extracted_data.json', 'w') as f:
        json.dump(processed_data, f)
        print("=========== File Processing Finished===========")
    return processed_data

def main():
  find_or_create_pinecode_index()
  data = extract_and_load_info_file()
  load_and_get_embeddings(data)
  # Step 3: Process a user query
  question = "What are the payment terms for deal 1?"
  question_embedding = get_embeddings(question)

  # Step 4: Query the vector database to retrieve relevant documents
  deal_id = 1
  top_docs = query_pinecone(deal_id, question_embedding)

  # Step 5: Combine top documents into a single context
  # context = " ".join([doc['matches'][0]['metadata']['content'] for doc in top_docs])

  # Step 6: Use Llama to answer the question based on retrieved documents
  answer = ask_llama(question, {})

  # Step 5: Add the reference URLs to the output
  references = "\nReferences:\n"
  for doc in top_docs:
    references += f"- [Document Link]({doc['url']})\n"

  # Return the full response, including references and the Llama answer
  print(answer + references)
  return answer + references


__main__ = main()

