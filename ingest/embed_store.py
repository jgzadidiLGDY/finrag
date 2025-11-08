import faiss, numpy as np, pandas as pd
from openai import OpenAI
from config import ARTIFACTS_DIR
from ingest.pdf_to_chunks_tokens import pdf_to_chunks
import os
BATCH=64

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def embed_texts(texts, model="text-embedding-3-small"):
    """
    This function converts each text in texts to a normalized vector (embeddings).
      ... instead of one text one API call a time, it does one batch o texts one API a time. 
    """    
    client = OpenAI()
    out = []
    for i in range(0, len(texts), BATCH):
        # build batch of texts
        batch = texts[i:i+BATCH]
        # one batch o texts one API
        resp = client.embeddings.create(model=model, input=batch)  
        out.extend([d.embedding for d in resp.data])
    X = np.array(out, dtype="float32")
    # norm, so dot product is essentially cosine similarity
    faiss.normalize_L2(X)
    return X

def embed_texts_single(texts, model="text-embedding-3-small"):
    """
    This function converts each text in texts to a normalized vector (embeddings).
    ... one text one API call a time
    """    
    client = OpenAI()
    # batch to avoid long payloads
    out = []
    for t in texts:
        out.append(client.embeddings.create(model=model, input=t).data[0].embedding)
    X = np.array(out, dtype="float32")
    # norm, so dot product is essentially cosine similarity
    faiss.normalize_L2(X)
    return X

def build_index(df, index_path="finrag.index", meta_path="finrag_meta.parquet"):
    """
    This function does followings 
    (1) convert df["text"] to vectors/embeddings, df is Pandas DataFrame
    (2) creates FAISS search index
    (3) saves the search index (incl. meta data) locally 
    """        
    X = embed_texts(df["text"].tolist())
    index = faiss.IndexFlatIP(X.shape[1]) 
    index.add(X)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    faiss.write_index(index, f"{ARTIFACTS_DIR}/{index_path}")
    df.to_parquet(f"{ARTIFACTS_DIR}/{meta_path}", index=False)
    return index.ntotal

def index_pdf(pdf_path, ticker, year, doc_type):
    """
    This function does followings 
    (1) splits a pdf to chunks
    (2) convets those chunks to vectors/embeddings
    (3) creates FAISS search index
    (4) saves the search index (incl. meta data) locally 
    """     
    rows = list(pdf_to_chunks(pdf_path, ticker, year, doc_type))
    texts = [r["text"] for r in rows]
    X = embed_texts_single(texts)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, f"{ARTIFACTS_DIR}/finrag.index")
    meta = pd.DataFrame(rows)
    meta.to_parquet(f"{ARTIFACTS_DIR}/finrag_meta.parquet", index=False)
    print("added vectors:", X.shape[0])

#if __name__ == "__main__":
#    index_pdf("data/raw/AAPL_10K_2024.pdf","AAPL",2024,"10-K")
