# retrieval/keyword.py
"""this code defines a simple keyword-based retrieval system using the BM25 (Best Matching 25) ranking function. 
    > it works as a component in the larger finRAG) pipeline. 
    > it uses BM25 scoring formulas 
"""
import math, re, json
from collections import Counter
from pathlib import Path  #

TOKEN = re.compile(r"[A-Za-z0-9_.$%-]+")  # all alphanumeric

# takes a string, finds all alphanumeric tokens, and returns them as a list of lowercase strings
def _tokenize(text: str):
    return [t.lower() for t in TOKEN.findall(text)]

# this class handles the indexing and searching of a collection of documents.
class MiniBM25:
    def __init__(self, docs):  # docs = list[str]
        """
        initialized with a list of documents (docs). It performs the following setup. 
          > tokenizes and converts all documents to lowercase using a regular expression.
          > calculates and stores several statistics needed for BM25 ranking 
          > sets the k1 and b parametersthe, ie., term frequency scaling and document length normalization
        """       
        self.docs = docs
        self.N = len(docs)
        self.doc_lens = [len(_tokenize(d)) for d in docs]
        self.avgdl = sum(self.doc_lens)/max(1,self.N)
        self.df = Counter()
        self.tfs = []
        for d in docs:
            toks = _tokenize(d)
            tf = Counter(toks); self.tfs.append(tf)
            for w in set(toks): self.df[w]+=1
        self.k1, self.b = 1.5, 0.75

    def _idf(self, w):
        """
        it calculates the Inverse Document Frequency (IDF) for a given word. 
          > the IDF measures how rare or common a word is across all documents. 
          > A higher IDF value means a word is more unique and thus more important for ranking.
        """          
        df = self.df.get(w, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, q, i):
        """
        it calculates the BM25 score for a given query and a specific document index i. 
           > it uses relevant BM25 formulas 
        """          
        # q: query string, i: doc index
        toks = _tokenize(q)
        tf = self.tfs[i]
        dl = self.doc_lens[i]
        score = 0.0
        for w in set(toks):
            if w not in tf: continue
            idf = self._idf(w)
            num = tf[w]*(self.k1+1)
            den = tf[w] + self.k1*(1 - self.b + self.b*dl/self.avgdl)
            score += idf * (num/den)
        return score

    def search(self, q, k=10):
        """
        it takes a query and returns the top k most relevant documents. 
        """             
        scores = [(i, self.score(q, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# the functionp loads the corpus of documents from a chunks.json file.
def load_corpus_texts(meta_path="data/artifacts/finrag_meta.parquet", cache_json="data/artifacts/chunks.json"):
    # If you already have a list[str] of chunk texts in a JSON, use it.
    if Path(cache_json).exists():
        return json.loads(Path(cache_json).read_text(encoding="utf-8"))
    # Otherwise, adapt this ... 
    raise FileNotFoundError("Provide chunks.json (list of chunk texts) extracted from finrag_meta.parquet")
