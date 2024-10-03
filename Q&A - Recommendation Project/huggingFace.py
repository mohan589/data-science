from datasets import load_dataset
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

raw_datasets = load_dataset('squad')
context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]

inputs = tokenizer(
  question,
  context,
  max_length=100,
  truncation="only_second",
  stride=50,
  return_overflowing_tokens=True,
  return_offsets_mapping=True)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))