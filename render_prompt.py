def render(bundle):
    """
    it builds a prompt with combined data of semantic retrieval and SQL-query  
      > it works with fuse.py, which provides bundle input 
      > it specifically lists citations of structured and/or unstructured sources

    for example, 
        === SOURCES (Unstructured) ===
        [U1] NVDA_10Q_2024Q2: para 118
        <chunk text…>

        [U2] AAPL_10K_2024: p.37
        <chunk text…>

        === SOURCES (Structured) ===
        [S1] AAPL | 2024 | Q2 | revenue=90800000000
        [S2] AAPL | 2024 | Q1 | revenue=117000000000
    """   
    hits = bundle.get("faiss_hits", []) or []
    fin = (bundle.get("financials") or {}).get("data", []) or []
    parts = []
    parts.append("You will answer a financial question using ONLY the sources listed below.")
    parts.append("Use bracket citations like [U1] or [S1] after each claim.")
    parts.append("")
    parts.append("=== QUESTION ===")
    parts.append(bundle.get("query",""))
    parts.append("")
    parts.append("=== SOURCES (Unstructured) ===")
    for i, h in enumerate(hits, 1):
        meta = h.get("meta", {}) or {}
        ident = meta.get("id") or meta.get("source") or "UNK"
        loc = meta.get("page") or meta.get("loc") or ""
        parts.append(f"[U{i}] {ident}{(' p.'+str(loc)) if loc else ''}")
        parts.append((h.get("text") or "").strip())
        parts.append("")
    parts.append("=== SOURCES (Structured) ===")
    for j, row in enumerate(fin, 1):
        # also store 'sid' = f"S{j}" into the row if you need it downstream
        parts.append(f"[S{j}] {row.get('ticker')} | {row.get('year')} | {row.get('quarter')} | "
                     f"{row.get('metric')}={row.get('value')}")
    return "\n".join(parts)
