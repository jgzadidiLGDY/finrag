# retrieval/search.py
import faiss, numpy as np, pandas as pd
from openai import OpenAI
from config import ARTIFACTS_DIR

def _embed_query(q, model="text-embedding-3-small"):
    """
    the functionpPerforms the actual embedding logic (internal helper).
      > converts each query a normalized vector (embeddings)
      > not part of the public API and should not be called directly.
    """  
    client = OpenAI() 
    v = client.embeddings.create(model=model, input=q).data[0].embedding
    v = np.array([v], dtype="float32")
    faiss.normalize_L2(v)  #normalized vector 
    return v

def embed_query(q, model="text-embedding-3-small"):
    """
    Public function to get an embedding for a query.
    Performs validation and other checks before calling the internal helper.
    """  
    if not isinstance(q, str):
        raise TypeError("Query must be a string.")
    
    # delegate the work to the internal function
    embedding = _embed_query(q, model=model)
    return embedding

def search(query, k=8, filters=None):
    """
    Performs a search with optional parameters.
    :param query: The search query string.
    :param k: The number of results to return. Defaults to 8.
    :param filters: A list of filters to apply to the search. Defaults to None.
    """    
    idx = faiss.read_index(f"{ARTIFACTS_DIR}/finrag.index")
    meta = pd.read_parquet(f"{ARTIFACTS_DIR}/finrag_meta.parquet")
    v = _embed_query(query)
    D, I = idx.search(v, k*3)  # overfetch
    rows = meta.iloc[I[0]].copy() 
    rows["score"] = D[0]
    if filters:
        for key, val in filters.items():
            rows = rows[rows[key] == val]
    return rows.nlargest(k, "score")[["score","ticker","year","doc_type","section","page","text"]]
    #return rows.nlargest(k, "score")[["score","ticker","year","doc_type","page","text"]]


def search_old(query, k=5):
    """
    This function does followings 
    (1) load index and meta from saved index 
    (2) call embed_query()convets query to vectors/embeddings
    (3) search through index 
    (4) process and return top-k searches
    """  
    index = faiss.read_index(f"{ARTIFACTS_DIR}/finrag.index")
    meta = pd.read_parquet(f"{ARTIFACTS_DIR}/finrag_meta.parquet")
    v = embed_query(query) 
    D, I = index.search(v, k)  # search 'v' in the index, return top-k  
    hits = []
    for rank, idx in enumerate(I[0]):
        rec = meta.iloc[int(idx)].to_dict() 
        rec["score"] = float(D[0][rank]) 
        hits.append(rec)
    return hits

#if __name__ == "__main__":
#    for h in search_old("summarize risk factors", k=3):
#        print(f"[{h['score']:.3f}] p.{h['page']} {h['doc_type']} {h['ticker']} :: {h['text'][:120]}…")
