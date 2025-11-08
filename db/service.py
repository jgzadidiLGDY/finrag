from db.queries import get_eps, get_revenue

def quarters_order(qs): 
    """
    This function sorts a list of strings representing financial quarters into chronological order. 
    """  
    order = {"Q1":1,"Q2":2,"Q3":3,"Q4":4} 
    #sorted() built-in function
    return sorted(qs, key=lambda q: order[q])

def get_eps_and_rev(ticker: str, fiscal_year: int, quarters: list[str]):
    """
    This function fetches and format financial data (earnings per share and revenue) 
    for a specified company and fiscal year, ensuring the output is ordered by quarter
    """  
    qs = quarters_order([q.upper() for q in quarters])
    eps = dict(get_eps(ticker, fiscal_year, qs))
    rev = dict(get_revenue(ticker, fiscal_year, qs))
    #  a list of dictionaries, one for each quarter
    data = [{"quarter": q, "eps": eps.get(q), "revenue": rev.get(q)} for q in qs]
    #a structured, complete output
    return {"ticker": ticker.upper(), "fiscal_year": fiscal_year, "data": data}

# a helper inspects rows and, for matching metrics (e.g. revenue), 
# computes QoQ Change
def compute_insight(financials, field="revenue", q1_label="Q1", q2_label="Q2", cite_tag="S1"):
    """
    Accepts either:
      - wide rows: [{'quarter':'Q1','revenue':..., 'eps':...}, ...]
      - long rows: [{'metric':'revenue','quarter':'Q1','value':...}, ...]
    Returns: (label, value_str, cite_tag) or None
    """
    if not financials:
        return None

    # Unwrap dict{'data': [...]} if needed
    if isinstance(financials, dict) and "data" in financials:
        rows = financials["data"]
    else:
        rows = financials

    # Normalize to a quarter -> value map for the requested field
    qval = {}
    # Wide shape
    for r in rows:
        if isinstance(r, dict) and "quarter" in r and field in r:
            q = str(r["quarter"]).upper()
            try:
                qval[q] = float(str(r[field]).replace(",", ""))
            except Exception:
                pass
    # Long shape fallback
    if not qval:
        for r in rows:
            if isinstance(r, dict) and str(r.get("metric","")).lower() == field:
                q = str(r.get("quarter","")).upper()
                try:
                    qval[q] = float(str(r.get("value")).replace(",", ""))
                except Exception:
                    pass

    q1, q2 = qval.get(q1_label.upper()), qval.get(q2_label.upper())
    if q1 is None or q2 is None or q1 == 0:
        return None
    
    # computes QoQ Change
    delta_pct = (q2 - q1) / abs(q1) * 100.0
    return (f"{field.capitalize()} QoQ", f"{delta_pct:+.1f}%", cite_tag)