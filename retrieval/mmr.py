# retrieval/mmr.py
import numpy as np
from typing import List, Dict

# calculates the cosine similarity between two vectors, a and b, using the NumPy library
def _cos(a, b, eps=1e-9):
    na = np.linalg.norm(a) # L2 norm
    nb = np.linalg.norm(b)
    # handle the zero-vector case
    if na*nb < eps: 
        return 0.0
    return float(np.dot(a,b)/(na*nb))

def maximal_marginal_relevance(query_vec: np.ndarray,
                               doc_vecs: np.ndarray,
                               k: int = 6,
                               lambda_relevance: float = 0.7) -> List[int]:
    """
    this function implements the Maximal Marginal Relevance (MMR) algorithm to diversify search results,
     balancing relevance to the query with dissimilarity among selected items. The core logic involves 
     iteratively selecting documents based on a weighted score that considers both their similarity to 
     the query and their dissimilarity to already chosen documents.
    """
    n = doc_vecs.shape[0]
    if n == 0: 
        return []
    sims = np.array([_cos(query_vec, doc_vecs[i]) for i in range(n)])
    selected, candidate_ids = [], set(range(n))
    # to build a list of diverse search results.
    while len(selected) < min(k, n): # It runs until a sufficient number of results (k) have been chosen
        mmr_scores = []
        for i in candidate_ids:
            if not selected:    # handle the very first document
                div = 0.0
            else:
                div = max(_cos(doc_vecs[i], doc_vecs[j]) for j in selected)
            mmr = lambda_relevance * sims[i] - (1 - lambda_relevance) * div
            mmr_scores.append((i, float(mmr)))
        # select the best candidate
        i_star = max(mmr_scores, key=lambda x: x[1])[0]  
        selected.append(i_star)
        candidate_ids.remove(i_star)
    # returns a list of integers representing the indices of the documents, sorted by their MMR score.    
    return selected

def apply_mmr_to_hits(query_vec, doc_vecs, hits: List[Dict], k=6, lam=0.7):
    """
     this function acts as a wrapper that takes the vectorized data and the original search results (hits), 
     uses the MMR logic to determine the best order, and then applies that order to the original list of dictionary-based hit objects
    """       
    order = maximal_marginal_relevance(query_vec, doc_vecs, k=k, lambda_relevance=lam)
    return [hits[i] for i in order]
