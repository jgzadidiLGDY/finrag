# sanity.py
from openai import OpenAI
import tiktoken

client = OpenAI()
enc = tiktoken.encoding_for_model("text-embedding-3-small")
print("token_count:", len(enc.encode("Hello FinRAG")))
e = client.embeddings.create(model="text-embedding-3-small", input="Hello FinRAG")
print("emb_dim:", len(e.data[0].embedding))