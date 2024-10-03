from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from transformers import BitsAndBytesConfig

import pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import os
import sys
from langchain import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA

loader = PyPDFDirectoryLoader("pdfs")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

os.environ.get('PINECONE_API_KEY', '5f00d642-14fd-4fd3-acb4-60f9976000ea')
os.environ.get('PINECONE_API_ENV', 'us-east-1')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '5f00d642-14fd-4fd3-acb4-60f9976000ea')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-east-1')

index = "langchainpinecone" # put in the name of your pinecone index here


docsearch = PineconeVectorStore(index=index, pinecone_api_key=PINECONE_API_KEY, embedding=embeddings).from_texts([t.page_content for t in docs], embeddings, index_name=index)
# print(docsearch, 'docsearch')

query = "YOLOv7 outperforms which models"

docs = docsearch.similarity_search(query, k=4)

import torch
tokenizer = AutoTokenizer.from_pretrained("/Users/mpichikala/personal/Llama-2")
model = AutoModelForCausalLM.from_pretrained("/Users/mpichikala/personal/Llama-2",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             # load_in_8bit=True,
                                             # quantization_config=BitsAndBytesConfig()
                                             )
pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id)
llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.1})
SYSTEM_PROMPT = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer."""
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<>\n", "\n<>\n\n"

SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS
instruction = """
{context}

Question: {question}
"""

template = B_INST + SYSTEM_PROMPT + instruction + E_INST
# print(template)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
# result = qa_chain("YOLOv7 is used for")
# print(result['result'])

while True:
  user_input = input(f"prompt:")
  if user_input == 'exit':
    print('Exiting')
    sys.exit()
  if user_input == '':
    continue
  result = qa_chain({'query': user_input})
  print(f"Answer:{result['result']}")
