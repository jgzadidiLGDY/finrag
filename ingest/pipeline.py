# ingest/pipeline.py
from ingest.pdf_to_chunks_tokens import pdf_to_chunks
import pandas as pd, os
from config import ARTIFACTS_DIR
#from sections import detect_section
from ingest.sections import detect_section

def ingest_pdf_to_rows(pdf_path, ticker, year, doc_type):
    """
    This function turn a list int pandas DataFrame, which is an excel-like data structure. 
    """      
    rows = list(pdf_to_chunks(pdf_path, ticker, year, doc_type))
    df_rows = pd.DataFrame(rows)
    # tagging "section"
    df_rows["section"] = df_rows["text"].apply(detect_section)
    return df_rows

def save_rows(df, name="chunks.parquet"):
    """
    saving data, simple. 
    """    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out = os.path.join(ARTIFACTS_DIR, name)
    df.to_parquet(out, index=False)
    return out

#if __name__ == "__main__":
#    df = ingest_pdf_to_rows("data/raw/AAPL_10K_2024.pdf","AAPL",2024,"10-K")
#    print(df.head())
#    print("rows:", len(df))
