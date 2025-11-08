from db.service import get_eps_and_rev
from retrieval.search import search

def _normalize_hits(hits):
    """
    a utility function designed to standardize a list of search results (hits) into a consistent, predictable format. 
    This is a common practice in data processing pipelines, especially in systems that aggregate data from 
    multiple sources or handle varied input format. 
      > it iterates through each item in the input list and transforms it into a dictionary with a specific set of keys
    """   
    norm = []
    for i, h in enumerate(hits, start=1):
        if isinstance(h, str):
            # a single, uniform format.
            norm.append({"id": i, "text": h, "score": None, "meta": {}})
        else:
            # tolerate various shapes
            norm.append({
                "id": h.get("id", i),
                "text": h.get("text", str(h)),  # a single, uniform format.
                "score": h.get("score", None),
                "meta": h.get("meta", {})
            })
    return norm

def combine_sources(query, ticker=None, year=None, quarters=None, k=6):
    """
    it combines data from two different sources: a retrieval-based search and a financial database
    """   
    raw_hits = search(query, k=k)
    hits = _normalize_hits(raw_hits)   # ← make dicts with 'text'
    fin = {}
    if ticker and year:
        fin = get_eps_and_rev(ticker, year, quarters or ["Q1","Q2","Q3","Q4"])
    return {"query": query, "faiss_hits": hits, "financials": fin}

#if __name__ == "__main__":
#    bundle = combine_sources("Summarize AAPL 2024 revenue", "AAPL", 2024, ["Q2"])
#    print(bundle)
