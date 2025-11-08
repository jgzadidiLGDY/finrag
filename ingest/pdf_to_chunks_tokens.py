import fitz, tiktoken
from typing import Dict, Generator, Optional

def _enc(model="text-embedding-3-small"):
    """
    tiktoken is OpenAI's tokenizer. essential for accurate chunking, token counting, and cost control.
    Tokenizer's job: 
      (1) map text → tokens → IDs (numbers)
      (2) map IDs (numbers) → tokens → text 
    """
    try: 
        return tiktoken.encoding_for_model(model)
    except KeyError: 
        return tiktoken.get_encoding("cl100k_base")


def chunk_by_tokens(text, enc, size=1000, overlap=150, min_chars=30):
    """
    This function will yield all overlapping chunks until it reaches the end of the text.
    """
    text = " ".join(text.split())
    toks = enc.encode(text)
    n = len(toks)
    step = max(1, size - overlap)
    for start in range(0, n, step):
        chunk = enc.decode(toks[start:min(n, start+size)]).strip()
        if len(chunk) >= min_chars: 
            yield chunk


def pdf_to_chunks(path:str, ticker:str, year:Optional[int], doc_type:str="10-K",
                  model="text-embedding-3-small", size=1000, overlap=150):
    """
    Extract text from a PDF and return list of dicts:
    [{ "page": i, "text": ..., "ticker": ..., "year": ..., "doc_type": ... }, ...]
    """
    enc = _enc(model)
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        raw = page.get_text("text") or ""; idx = 0
        for ch in chunk_by_tokens(raw, enc, size=size, overlap=overlap):
            yield {"page": i+1, "text": ch, "ticker": ticker, "year": year,
                   "doc_type": doc_type, "chunk_index": idx}
            idx += 1
