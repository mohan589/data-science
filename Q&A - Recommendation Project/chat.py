import fitz  # PyMuPDF
from transformers import AutoModel, AutoTokenizer
import torch
from pinecone import Pinecone, ServerlessSpec
import tensorflow as tf

# Check if MPS is available and use it; otherwise, fall back to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Step 1: Extract Text from PDFs
def extract_text_from_pdf(pdf_path):
  text = ""
  with fitz.open(pdf_path) as doc:
    for page in doc:
      text += page.get_text()
  return text

# Step 2: Embed the Text with LLaMA
tokenizer = AutoTokenizer.from_pretrained("/Users/mpichikala/personal/llama-2/")
model = AutoModel.from_pretrained("/Users/mpichikala/personal/llama-2/").to(device)

def embed_text(text):
  # Set padding token if not already set
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

  inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
  inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to MPS
  with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)
  return embeddings.squeeze().cpu().tolist()

pc = Pinecone(api_key="5f00d642-14fd-4fd3-acb4-60f9976000ea")
index_name = "chat"

def find_or_create_pinecode_index():
  # Step 3: Store Embeddings in Pinecone
  if index_name not in pc.list_indexes().names():
    pc.create_index(name='chat',
                          dimension=4096,  # Replace with your model dimensions
                          metric="cosine",  # Replace with your model metric
                          spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                          ))
# index = pc.Index("chat")

def store_embeddings(embeddings, metadata):
  pc.Index("chat").upsert(vectors=[{ 'id': metadata['id'], 'values': embeddings, 'metadata': metadata }], batch_size=2)

# Step 4: Process PDFs, Extract, Embed, and Store
def process_pdf(pdf_path):
  # Extract text from the PDF
  text = extract_text_from_pdf(pdf_path)

  # Embed the extracted text
  embeddings = embed_text(text)

  # Create metadata (e.g., document ID)
  metadata = {
    'id': pdf_path,  # Use the PDF path as an ID or generate a unique ID
    'file_name': pdf_path.split('/')[-1],  # Just an example
  }

  # Store embeddings in Pinecone
  store_embeddings(embeddings, metadata)

# Process your PDF files here
pdf_files = ["sample.pdf"]  # List of your PDF files
for pdf_file in pdf_files:
  process_pdf(pdf_file)

# Step 5: Query the Data
def query_embeddings(query):
  query_embedding = embed_text(query)

  # Ensure query_embedding is a flat list of floats
  if not isinstance(query_embedding, list) or not all(isinstance(i, float) for i in query_embedding):
    raise ValueError("Embedding must be a flat list of floats.")

  results = pc.Index("chat").query(vector=query_embedding, top_k=10, include_metadata=True,  include_values=True)
  return results

# Step 6: Generate a Response using LLaMA
def generate_response(context, user_query):
  prompt = f"User question: {user_query}\nContext: {context}\nResponse:"
  # Set padding token if not already set
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
  inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)

  with torch.no_grad():
    output = model.generate(**inputs, max_length=150)

  response = tokenizer.decode(output[0], skip_special_tokens=True)
  return response

def main():
  find_or_create_pinecode_index()
  # Step 7: Chat Interface
  while True:
    user_input = input("Ask your question: ")
    if user_input.lower() == "exit":
      break

    results = query_embeddings(user_input)

    # Collect context from the matched documents
    context = ""
    for match in results['matches']:
      context += f"Document ID: {match['id']}, Score: {match['score']}\n"
      # Optionally, add more details from the matched documents

    # Generate a response from LLaMA
    response = generate_response(context, user_input)
    print(f"LLaMA Response: {response}")

if __name__ == "__main__":
  main()