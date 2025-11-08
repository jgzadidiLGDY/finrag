# ingest/export_chunks.py
"""
Export chunk texts from finrag_meta.parquet into data/artifacts/chunks.json
so that keyword/BM25 retriever can index them.
  > it is a one-off function or use. 
"""
import pandas as pd
import json, os

META_PATH = "data/artifacts/finrag_meta.parquet"
OUT_PATH = "data/artifacts/chunks.json"

def export_chunks(meta_path=META_PATH, out_path=OUT_PATH, text_col="text"):
    """
    it reads a Parquet file, extracts a specific text column from it, and saves the contents of that column as
    a list of strings in a new JSON file. It includes several checks to ensure the process runs smoothly. 
       > it creates the data source that the MiniBM25 class uses
    """   
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found. Run ingestion pipeline first.")

    df = pd.read_parquet(meta_path)
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in {meta_path} columns: {df.columns.tolist()}")

    texts = df[text_col].astype(str).tolist()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)  #JSON export

    print(f"Exported {len(texts)} chunk texts to {out_path}")

if __name__ == "__main__":
    export_chunks()
