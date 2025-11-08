""" a utility class for generating vector embeddings from text using OpenAI's API. 
It acts as an adapter, providing a clean, encapsulated way to interact with the underlying API.
"""
from openai import OpenAI
import numpy as np

class Embedder:
    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def embed_query(self, text: str):
        """Return a single query vector (list of floats)."""
        res = self.client.embeddings.create(model=self.model, input=text)
        return np.array(res.data[0].embedding, dtype=float)

    def embed_documents(self, texts):
        """Return list of vectors for multiple texts."""
        res = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([r.embedding for r in res.data], dtype=float)
