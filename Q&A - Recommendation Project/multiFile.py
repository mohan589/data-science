import fitz  # PyMuPDF
import torch

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Example usage
pdf_path = 'hr.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
# print(pdf_text[:500])  # Print the first 500 characters of the PDF content

from sentence_transformers import SentenceTransformer
#
# # Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
#
# # Split text into smaller chunks for embedding (optional based on your content)
pdf_sections = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 1000)]  # Split by paragraphs
#
# # Embed each section
embeddings = model.encode(pdf_sections)
# print(f"Generated {len(embeddings)} embeddings from the PDF.")
#
from pinecone import Pinecone, ServerlessSpec
#
# # Initialize Pinecone
pc = Pinecone(api_key='5f00d642-14fd-4fd3-acb4-60f9976000ea')
#
# # pinecone.init(api_key='your-pinecone-api-key', environment='us-west1-gcp')
#
# # Create a new index (if not already created)
index_name = 'pdf-embeddings'
# if index_name not in pc.list_indexes():
#     pc.create_index(index_name, dimension=embeddings.shape[1], spec=ServerlessSpec(
#                             cloud="aws",
#                             region="us-east-1"
#                           ))
#
# # Connect to the Pinecone index
index = pc.Index(index_name)
#
# # Insert embeddings into Pinecone
ids = [str(i) for i in range(len(embeddings))]
# print(f"ids are {ids} and embeddings are {embeddings}")
index.upsert(vectors=zip(ids, embeddings))
#
# print(f"Stored {len(embeddings)} embeddings in Pinecone.")


def query_pinecone(query, top_k=10):
  # Encode the query
  query_embedding = model.encode([query], device=device)

  # Search in Pinecone
  results = index.query(vector=query_embedding.astype(float).tolist(), top_k=top_k)

  # Return matched text sections
  matches = results['matches']
  return [pdf_sections[int(match['id'])] for match in matches]

from transformers import pipeline

# Load a question-answering model
qa_model = pipeline('question-answering', model='deepset/roberta-base-squad2', device=device)

# Use the model to answer based on retrieved sections
def answer_based_on_pdf(query, matched_sections):
  context = " ".join(matched_sections)  # Combine the matched sections
  if context.strip() == "":
    return "Don't know"

  context = " ".join(matched_sections)  # Combine the matched sections for context
  if not context.strip():
    return "Don't know"

  result = qa_model(question=query, context=context)
  print(f"Answer: {result['answer']}")
  return result['answer'] if result['score'] > 0.0001 else "Don't know"

# from datasets import Dataset
# from transformers import RobertaTokenizer, RobertaForQuestionAnswering, Trainer, TrainingArguments
#
# # Load tokenizer and model
# tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")
# model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
#
# # Create custom dataset from your PDFs (in Q&A format)
# data = {
#     'id': ['what is employee count', 'what are strategies'],
#     'context': ['all about human resources in a corporate world', 'hr strategies'],
#     'question': ['what is employee count?', 'what are strategies?'],
#     'answers': [{'text': ['3500'], 'answer_start': [10]}, {'text': ['answer2'], 'answer_start': [25]}]
# }
#
# dataset = Dataset.from_dict(data)
#
# # Tokenize dataset
# def prepare_features(examples):
#     return tokenizer(
#       examples["question"],
#       examples["context"]
      # truncation="only_second",  # Truncate context if necessary
      # padding="max_length",
      # max_length=384,
      # return_overflowing_tokens=True  # Handle longer sequences
    # )
    #
    # sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # offset_mapping = tokenized_examples.pop("offset_mapping")
    #
    # tokenized_examples["start_positions"] = []
    # tokenized_examples["end_positions"] = []
    #
    # for i, offsets in enumerate(offset_mapping):
    #     # Get the sample that generated this example
    #     sample_index = sample_mapping[i]
    #     answers = examples["answers"][sample_index]
    #     start_char = answers["answer_start"][0]
    #     end_char = start_char + len(answers["text"][0])
    #
    #     # Find the start and end token positions in the context
    #     sequence_ids = tokenized_examples.sequence_ids(i)
    #
    #     # The context is the second part of the sequence, and we search for answer positions within the context
    #     context_start = sequence_ids.index(1)
    #     context_end = len(sequence_ids) - sequence_ids[::-1].index(1)
    #
    #     # Find the start and end token positions
    #     start_token_pos = context_start
    #     end_token_pos = context_end - 1
    #
    #     for idx in range(context_start, context_end):
    #         if offsets[idx][0] <= start_char and offsets[idx][1] >= start_char:
    #             start_token_pos = idx
    #         if offsets[idx][0] <= end_char and offsets[idx][1] >= end_char:
    #             end_token_pos = idx
    #
    #     tokenized_examples["start_positions"].append(start_token_pos)
    #     tokenized_examples["end_positions"].append(end_token_pos)
    #
    # return tokenized_examples

# tokenized_datasets = dataset.map(prepare_features, batched=True)
#
# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results', num_train_epochs=3, per_device_train_batch_size=4
# )
#
# # Initialize Trainer and train the model
# trainer = Trainer(
#     model=model, args=training_args, train_dataset=tokenized_datasets
# )
#
# trainer.train()


# Example query
query = "automotive industry employee count?"
matched_sections = query_pinecone(query)
# for i, section in enumerate(matched_sections, 1):
#   print(f"Match {i}: {section[:200]}")  # Print first 200 characters of each match

# Get the final response
response = answer_based_on_pdf(query, matched_sections)
print(f"Response: {response}")
