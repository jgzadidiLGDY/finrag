# scripts/build_corpus.py
from ingest.pipeline import ingest_pdf_to_rows, save_rows
from ingest.embed_store import build_index
import pandas as pd

# hard-cded those SDF files for now 
FILES = [
    ("data\\raw\\AAPL_10K_2024.pdf", "AAPL", 2024, "10-K"),
    ("data\\raw\\NVDA_10K_2024.pdf", "NVDA", 2024, "10-K"),
    ("data\\raw\\MSFT_10K_2024.pdf", "MSFT", 2024, "10-K"),
    ("data\\raw\\MSFT_10Q_2024Q1.pdf", "MSFT", 2024, "10-Q"),
    ("data\\raw\\MSFT_10Q_2024Q2.pdf", "MSFT", 2024, "10-Q"),
    ("data\\raw\\MSFT_10Q_2024Q3.pdf", "MSFT", 2024, "10-Q"),
    ("data\\raw\\AAPL_10Q_2024Q1.pdf", "AAPL", 2024, "10-Q"),
    ("data\\raw\\AAPL_10Q_2024Q2.pdf", "AAPL", 2024, "10-Q"),
    ("data\\raw\\AAPL_10Q_2024Q3.pdf", "AAPL", 2024, "10-Q"),
    ("data\\raw\\NVDA_10Q_2024Q1.pdf", "NVDA", 2024, "10-Q"),
    ("data\\raw\\NVDA_10Q_2024Q2.pdf", "NVDA", 2024, "10-Q"),
    ("data\\raw\\NVDA_10Q_2024Q3.pdf", "NVDA", 2024, "10-Q"),
    ("data\\raw\\MSFT_Transcript_2024Q2.pdf", "MSFT", 2024, "Transcript"),
    ("data\\raw\\NVDA_Transcript_2024Q2.pdf", "NVDA", 2024, "Transcript"),
    ("data\\raw\\AAPL_Transcript_2024Q2.pdf", "AAPL", 2024, "Transcript"),
]

# rebuild index from concatenated dfs 
dfs = [ingest_pdf_to_rows(*args) for args in FILES]
corpus = pd.concat(dfs, ignore_index=True)
save_rows(corpus, "corpus_chunks.parquet")
print("chunks: ", len(corpus))
print("indexed: ", build_index(corpus, "finrag.index", "finrag_meta.parquet"))