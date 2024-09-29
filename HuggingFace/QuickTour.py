from transformers import pipeline
import torch
from datasets import load_dataset, Audio

classifier = pipeline('sentiment-analysis', device='mps')
results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device='mps')
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column('audio', Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

result = speech_recognizer(dataset[:4]['audio'])
print(d['text'] for d in result)
print(result)