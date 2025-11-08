"""
Deterministic hybrid retrieval + header filtering + MMR diversity.

Public API (import from this module):
  - HybridRetriever
  - drop_headers
  - make_scope_filter  (alias of _make_scope_filter)
"""

from __future__ import annotations
import re
import unicodedata
from typing import Callable, Iterable, List, Tuple, Optional, Dict, Any

import numpy as np

# Project-local imports
from retrieval.keyword import MiniBM25, load_corpus_texts
from retrieval.search import search as semantic_search_raw
from fuse import _normalize_hits

# ----------------------------
# Scoring utilities (deterministic)
# ----------------------------

def _rank_normalize(pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """
    Rank-based normalization to [0,1].
      > using positions to normalize scores 
    Input: [(doc_id, raw_score), ...] with higher better.
    Output: [(doc_id, norm in [0,1]), ...] top rank=1.0, last=0.0
    """
    if not pairs:
        return []
    ranked = sorted(pairs, key=lambda x: x[1], reverse=True)
    n = len(ranked)
    if n == 1:
        return [(ranked[0][0], 1.0)]
    return [(doc_id, 1.0 - i / (n - 1)) for i, (doc_id, _) in enumerate(ranked)]

def _stable_union_ids(list_a: List[Tuple[int, float]],
                      list_b: List[Tuple[int, float]]) -> List[int]:
    """
    it merges the two candidate lists — one from BM25, one from the semantic retriever 
      > preserving a deterministic order, BM25’s ordering first. 
      > removing duplicates (duplicate doc_ids).
    """
    seen, out = set(), []
    for i, _ in list_a:
        if i not in seen:
            seen.add(i); out.append(i)
    for i, _ in list_b:
        if i not in seen:
            seen.add(i); out.append(i)
    return out

# ----------------------------
# Robust header / boilerplate filter, which are basica formatting or non-relevant distraction
# _ALWAYS_DROP_ANYWHERE	Universal formatting boilerplate (“Form 10-K”, “SECURITIES AND EXCHANGE”)
# _HEAD_DROP	Section headers within the first ~800 chars	head-limited substring match
# _is_shouty	VISUAL all-caps headers (no keywords)	uppercase ratio heuristic
# ----------------------------

def _canon(s: str) -> str:
    """Casefold + normalize unicode + collapse whitespace + dash normalization."""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s

# ALWAYS drop if these appear anywhere ... no real data/info
_ALWAYS_DROP_ANYWHERE = (
    "united states securities",               # SEC cover
    "table of content",                       # catches contents/linebreak splits
    "indicate by check mark whether",
    "available information",
)

# Drop if these appear in the head region (true header-ish phrases)  ... no real data/info
_HEAD_DROP = (
    "forward looking statement",              # forward-looking statements
    "this annual report on form 10-k",
    "this quarterly report on form 10-q",
    "form 10-k",                              # head-limited to avoid nuking body cites
    "form 10-q",
)

def _is_shouty(head: str, min_letters: int = 80, min_upper_ratio: float = 0.45) -> bool:
    letters = [c for c in head if c.isalpha()]
    if len(letters) < min_letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= min_upper_ratio

# drop_headers() = composite, deterministic content-filter
# (_ALWAYS_DROP_ANYWHERE + _HEAD_DROP + _is_shouty)
# It keeps hybrid search and MMR results focused on real paragraphs, not filing structure or metadata.
def drop_headers(hit: Dict[str, Any]) -> bool:
    """
    Hard-filter: return False to exclude header/boilerplate chunks.
    Uses 'anywhere' markers, head-limited markers, and a shouty-cover heuristic.
    """
    text: str = hit.get("text", "")
    head = text[:800]

    t = _canon(text)
    h = _canon(head)

    if any(sub in t for sub in _ALWAYS_DROP_ANYWHERE):
        return False
    if any(sub in h for sub in _HEAD_DROP):
        return False
    if _is_shouty(head):
        return False
    return True

# ----------------------------
# Optional metadata scoping
# ----------------------------

try:
    import pandas as pd  # optional; only needed if you use scope filter
except Exception:  # pragma: no cover
    pd = None  # type: ignore

def _load_meta() -> Optional["pd.DataFrame"]: # type: ignore
    if pd is None:
        return None
    try:
        df = pd.read_parquet("finrag_meta.parquet")
        if "id" in df.columns:
            df = df.set_index("id")
        return df
    except Exception:
        return None

def _make_scope_filter(meta_df: Optional["pd.DataFrame"], *, # type: ignore
                       ticker: Optional[str] = None,
                       year: Optional[int] = None,
                       quarter: Optional[str] = None,
                       doctypes: Optional[Iterable[str]] = None,
                       drop_hdr: bool = True) -> Callable[[Dict[str, Any]], bool]:
    """
    Build a filter that enforces ticker/year/quarter/doctype using meta,
    while optionally also dropping headers via drop_headers.
    """
    allowed_doctypes = set(doctypes) if doctypes else None

    def _f(hit: Dict[str, Any]) -> bool:
        if drop_hdr and not drop_headers(hit):
            return False
        if meta_df is None:
            return True  # cannot scope without meta; header already filtered if requested
        i = hit.get("id", None)
        if i is None or i not in meta_df.index:
            return False
        row = meta_df.loc[i]
        if ticker is not None and str(row.get("ticker")) != str(ticker):
            return False
        if year is not None and int(row.get("year", 0)) != int(year):
            return False
        if quarter is not None and str(row.get("quarter")) != str(quarter):
            return False
        if allowed_doctypes is not None and str(row.get("doctype")) not in allowed_doctypes:
            return False
        return True

    return _f

# Public alias (clean name)
make_scope_filter = _make_scope_filter

# ----------------------------
# HybridRetriever
# ----------------------------

class HybridRetriever:
    """
    Hybrid retrieval leverages the strengths of two methods: BM 25 is precise for finding literal terms, 
    while semantic search is better for understanding the contextual meaning of a query.

    implementation wise: rank-normalized semantic + keyword blend with deterministic tie-breaks,
    optional hard-filter, and MMR diversification.

    alpha: weight on semantic channel in [0,1].
    semantic_is_distance: set True if semantic 'score' is actually a distance (e.g., L2).
    """

    def __init__(self, alpha: float = 0.7, semantic_is_distance: bool = False):
        # set up with two main retrieval components and a corpus of documents
        self.alpha = float(alpha)
        self.semantic_is_distance = bool(semantic_is_distance)

        self.docs: List[str] = load_corpus_texts()
        self.bm = MiniBM25(self.docs)
        self.meta = _load_meta()  # Optional DataFrame

    # ---- Semantic channels ----

    def _semantic(self, query: str, k: int = 50) -> Tuple[List[Tuple[int, float]], List[Dict[str, Any]]]:
        """
        Call the project semantic search, normalize the hits, and return (id, score) pairs.
        """
        raw_hits = _normalize_hits(semantic_search_raw(query, k=k))
        pairs: List[Tuple[int, float]] = []
        for idx, h in enumerate(raw_hits):
            doc_id = int(h.get("id", idx))
            score = float(h.get("score", 0.0) or 0.0)
            if self.semantic_is_distance:
                score = -score  # flip distances so higher is better, consistent with relvance 
            pairs.append((doc_id, score))
        return pairs, raw_hits

    # ---- main search ----

    def search(self, query: str, k: int = 10,
               filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        """
        The standard search combines the results from both the BM25 and semantic search channels to produce a final, blended ranking.
          1) (1-alpha) * BM25 + alpha*semantic
          2) rank-normalize each channel to a common scale
          3) stable union + convex combine (alpha)
          4) stable sort by (-score, doc_id)
          5) build hits, applying filter to remove undesirable hits
        """
        # Candidate pool (give filters room)
        pool_k = max(50, k * 8)

        # Channels
        bm_raw: List[Tuple[int, float]] = self.bm.search(query, k=pool_k)
        sem_pairs, _ = self._semantic(query, k=pool_k)

        bm_norm = dict(_rank_normalize(bm_raw))
        sem_norm = dict(_rank_normalize(sem_pairs))

        # Deterministic union
        candidate_ids = _stable_union_ids(bm_raw, sem_pairs)

        # Blend
        alpha = self.alpha
        combined: List[Tuple[int, float]] = []
        for i in candidate_ids:
            s = sem_norm.get(i, 0.0)
            b = bm_norm.get(i, 0.0)
            # combined scoring, the hybrid score 
            score = alpha * s + (1.0 - alpha) * b 
            combined.append((i, score))

        # Stable sort with tie-break on doc_id
        combined.sort(key=lambda t: (-t[1], t[0]))

        # Build hits (apply filter before counting to k)
        hits: List[Dict[str, Any]] = []
        for doc_id, score in combined:
            text = self.docs[doc_id] if 0 <= doc_id < len(self.docs) else ""
            hit = {
                "id": int(doc_id),
                "text": text,
                "score": float(score),  # hybrid score
                "meta": {"source": "hybrid", "alpha": alpha}
            }
            # stash hybrid score explicitly for downstream (MMR)
            hit["meta"]["hybrid_score"] = hit["score"]

            if filter_fn is None or filter_fn(hit):
                hits.append(hit)
                if len(hits) >= k:
                    break

        return hits

    # ---- MMR (diversity) ----

    def search_mmr(self, query: str, embedder, k: int = 6, lam: float = 0.8,
                   filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        """
        1. first runs the standard hybrid search (self.search) to get an initial set of highly relevant documents.
        2. re-ranks the top stardard search results to promote diversity, the MMR formula is 
           > MMR with relevance = hybrid score (from search) and diversity = doc–doc cosine.
        3. Greedy selection with deterministic tie-breaks.
           > The first document picked is the one with the highest hybrid score (relevance) from the initial candidate set.
           > Subsequent documents are chosen by balancing relevance (based on hybrid score) with diversity (dissimilarity to 
            already-selected documents),controlled by the lam parameter.
        """
        # Candidate set already hybrid-ranked + filtered
        cand = self.search(query, k=max(40, k), filter_fn=filter_fn)
        if not cand:
            return []

        # Embeddings for diversity only
        qv = np.asarray(embedder.embed_query(query), dtype=np.float32)
        dv = np.asarray(embedder.embed_documents([c["text"] for c in cand]), dtype=np.float32)

        # Unit-normalize
        qn = float(np.linalg.norm(qv)) or 1.0
        qv = (qv / qn).astype(np.float32)
        for i in range(dv.shape[0]):
            n = float(np.linalg.norm(dv[i])) or 1.0
            dv[i] = dv[i] / n

        # Relevance from hybrid score
        rel = np.array([c["meta"].get("hybrid_score", c["score"]) for c in cand], dtype=np.float32)

        # Deterministic MMR
        selected: List[int] = []
        picked = np.zeros(len(cand), dtype=bool)
        doc_ids = np.array([c["id"] for c in cand], dtype=np.int64)

        # First: highest rel with tie-break on doc_id
        first = np.lexsort((doc_ids, -rel))[0]
        selected.append(first)
        picked[first] = True

        # subsequent greedy selection process
        while len(selected) < min(k, len(cand)):
            # max cosine similarity to any selected item
            max_div = np.full(len(cand), -1.0, dtype=np.float32)
            for j in range(len(cand)):
                if picked[j]:
                    continue
                sims = float(np.dot(dv[j], dv[selected].T)) if len(selected) == 1 else np.dot(dv[j], dv[selected].T)
                if np.ndim(sims) == 0:
                    sims = np.array([sims], dtype=np.float32)
                max_div[j] = np.max(sims)

            mmr = lam * rel - (1.0 - lam) * max_div
            mmr[picked] = -1e9  # never pick already picked

            # choose next: max mmr, tie-break on doc_id
            nxt = np.lexsort((doc_ids, -mmr))[0]
            selected.append(nxt)
            picked[nxt] = True

        # Build results in selection order
        mmr_hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(selected, 1):
            h = dict(cand[idx])  # shallow copy
            h["meta"] = dict(h.get("meta", {}))
            h["meta"]["mmr"] = True
            h["meta"]["mmr_rank"] = rank
            # store relevance portion for visibility
            h["meta"]["mmr_score"] = float(lam * rel[idx])
            mmr_hits.append(h)

        # a list of documents that are both relevant to the query and varied in their content, 
        # reducing the chance of redundant information being returned
        return mmr_hits
