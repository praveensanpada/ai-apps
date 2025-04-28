import faiss
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average over token embeddings
    return embeddings

def store_embeddings(texts, index_path="knowledge_base/embeddings/faiss.index"):
    embeddings = generate_embeddings(texts).numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    index.add(embeddings)
    faiss.write_index(index, index_path)

def retrieve_answer(query, index_path="knowledge_base/embeddings/faiss.index", k=1):
    query_embedding = generate_embeddings([query]).numpy()
    index = faiss.read_index(index_path)
    D, I = index.search(query_embedding, k)
    return I  # Index of the top-k most similar answers
