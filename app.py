""" a Streamlit frontend web application that demonstrates a Hybrid Retrieval system, designed for
answering queries based on a corpus of financial documents. The system uses a 
Retrieval-Augmented Generation (RAG) approach to retrieve and produce relevant information and 
presents it to the user through an interactive interface, which features a main display area 
and a sidebar for controls.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import re
import json
# project imports 
from hybrid import HybridRetriever, drop_headers, make_scope_filter
from ingest.embedder import Embedder 
from llm_answer import llm_answer
from db.service import compute_insight
from fuse import combine_sources

st.set_page_config(page_title="FinRAG", layout="wide")
st.title("💡 FinRAG — Hybrid Retrieval Demo")

# ---------------- UI controls ----------------
# controling the importance of the semantic versus keyword search (alpha) and the diversity of results (lam)
# ---------------- UI controls ----------------
with st.sidebar:
    st.subheader("Retrieval Settings")
    query = st.text_input("Query", value="AAPL Q2 AI demand", key="query")
    k = st.slider("Top-k", 3, 12, 6, key="k")
    use_hybrid = st.toggle("Use hybrid (keyword + semantic)", value=True, key="use_hybrid")
    use_mmr = st.toggle("Apply MMR diversity (MMR)", value=True, key="use_mmr")
    alpha = st.slider("Hybrid α (semantic weight)", 0.0, 1.0, 0.70, 0.05, key="alpha")
    lam = st.slider("MMR λ (relevance vs diversity)", 0.0, 1.0, 0.80, 0.05, key="lam")
    st.divider()
    st.subheader("Filters")
    hide_headers = st.toggle("Hide headers/boilerplate", value=True, key="hide_headers")
    scope_toggle = st.toggle("Scope by Ticker/Year/Docs", value=False, key="scope_toggle") 
    # scope_toggle = False

    st.divider()
    show_json = st.toggle("Show JSON artifact", value=False, key="show_json")

# ---- make the buttons context-aware ----
# stash search results in st.session_state and disable the LLM button
# until there are hits (or until a successful combine_sources completes).
if "hits" not in st.session_state:
    st.session_state.hits = []
if "ready_for_llm" not in st.session_state:
    st.session_state.ready_for_llm = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Reset readiness when query text changes
if st.session_state.last_query != query:
    st.session_state.ready_for_llm = False
    st.session_state.hits = []
    st.session_state.last_query = query

# Compose filter_fn
filt = None
if scope_toggle:
    # combines metadata scoping + (optional) header drop
    # hr.meta is loaded inside HybridRetriever, so we create hr first then gate
    pass

# ---------------- init retriever/embedder ----------------
# Create retriever with new alpha (the so-called deterministic fusion)
hr = HybridRetriever(alpha=alpha)

# ---- Ensure hr.meta is loaded, normalized, and safe ----
# 1) Load meta if hr.meta is None
if getattr(hr, "meta", None) is None:
    try:
        meta_path = Path(__file__).parent / "data" / "artifacts" / "finrag_meta.parquet"
        hr.meta = pd.read_parquet(meta_path)
    except Exception as e:
        hr.meta = None
        st.sidebar.error(f"Metadata not loaded: {e}")

# 2) If it loaded, normalize column name doc_type -> doctype (no-op if already correct)
if isinstance(hr.meta, pd.DataFrame):
    if "doc_type" in hr.meta.columns and "doctype" not in hr.meta.columns:
        hr.meta = hr.meta.rename(columns={"doc_type": "doctype"})
    #st.sidebar.caption(f"meta rows={len(hr.meta)}, cols={list(hr.meta.columns)}")
else:
    st.sidebar.info("Scope controls disabled: metadata unavailable.")
    scope_toggle = False

# lines to normalize the column name ---
if "doc_type" in hr.meta.columns and "doctype" not in hr.meta.columns:
    hr.meta = hr.meta.rename(columns={"doc_type": "doctype"})

# load meta data to setup scope dropdown 
if getattr(hr, "meta", None) is None:
    try:
        meta_path = Path("data/artifacts/finrag_meta.parquet") 
        hr.meta = pd.read_parquet(meta_path)
        st.sidebar.success(f"Loaded metadata: {len(hr.meta)} rows")
    except Exception as e:
        st.sidebar.error(f"Could not load metadata from {meta_path}: {e}")

# define filters to get rid of formatting information in PDF files 
filt = drop_headers if hide_headers else None
if scope_toggle and hasattr(hr,"meta") and hr.meta is not None and not hr.meta.empty:
    tkr = st.sidebar.selectbox("Ticker", sorted(hr.meta["ticker"].dropna().unique().tolist()))
    years = sorted(set(int(y) for y in hr.meta.loc[hr.meta["ticker"]==tkr,"year"].dropna().tolist()), reverse=True)
    yr = st.sidebar.selectbox("Year", years)
    docts = sorted(hr.meta["doctype"].dropna().unique().tolist())
    sel = set(st.sidebar.multiselect("Doc Types", docts, default=[d for d in ("10-K","10-Q") if d in docts]))
    gate = make_scope_filter(hr.meta, ticker=tkr, year=int(yr), doctypes=sel or None, drop_hdr=hide_headers)
    filt = gate

# Optional embedder for MMR
embedder = None
if use_mmr:
    try:
        embedder = Embedder(model="text-embedding-3-small")
    except Exception as e:
        st.sidebar.warning(f"Embedder unavailable, falling back to hybrid only. ({e})")
        use_mmr = False


# ---------------- run retrieval ----------------
# the "Search" button is hit, the application executes the retrieval process based on the selected settings.
# it first initializes a HybridRetriever and potentially an Embedder for the semantic search part. (above)
# it then constructs a filter function (filt) based on the user's choices (e.g., to only search AAPL 2024 filings). (above)
# finally, it calls the hr.search_mmr() or hr.search() method to perform the search and get the top k relevant hits.
#
# the "LLM Answer" button is hit, the application connects to LLM model and sends a preparted prompt to the OpenAI API, 
# then processes and displays the returns from LLM, including latency.
# ---------------- 

hits = []
if st.button("Search", type="primary", key="btn_search"):
    if not query.strip():
        st.warning("Enter a query to search.")
    else:
        if use_hybrid:
            if use_mmr and embedder is not None:
                hits = hr.search_mmr(query, embedder=embedder, k=k, lam=lam, filter_fn=filt)
            else:
                hits = hr.search(query, k=k, filter_fn=filt)
        else:
            # if ever expose pure-BM25 or pure-semantic toggles, call those here
            hits = hr.search(query, k=k, filter_fn=filt)

        st.session_state.hits = hits
        st.session_state.ready_for_llm = len(hits) > 0
        st.success(f"Returned {len(hits)} hits")

# ---- Analist Report button (context-aware) ----
llm_disabled = not st.session_state.get("ready_for_llm", False)

col1, col2 = st.columns([1,3])
with col1:
    clicked1 = st.button("Generate Analyst Report", type="primary",
                        key="btn_llm", disabled=llm_disabled)
if clicked1:
    with st.spinner("Generating detailed analysis…"):
        # Use whatever scope you prefer; here’s a simple example:
        res = llm_answer(query, ticker="AAPL", year=2024, quarters=["Q2"])
        st.subheader("Analyst Report")
        st.write(res["answer"])
        st.caption(f"Latency: {res['latency_ms']:.0f} ms • Model: {res['model']}")
#if llm_disabled:
#    st.caption("Run a search first to enable Analyst Report.")

# --- Summarize in 5 bullets (consistent with Analyst Report pipeline) ---
sum_disabled = not st.session_state.get("ready_for_llm", False)
with col2:
    clicked2 = st.button("Quick 5-Bullet Summary",  type="primary",
                         key="btn_sum", disabled=sum_disabled)
if clicked2:
    with st.spinner("Summarizing key points…"):
        # minimal wrapper: add a task-specific instruction to the query, reuse llm_answer()
        task = ("Summarize the most relevant points in exactly 5 concise bullets. "
                "Focus on revenue, EPS, guidance, demand signals, and risks. "
                "Include citations with [U#]/[S#].")
        res = llm_answer(query + "\n\nTASK: " + task)

        st.subheader("5-Bullet Summary")
        st.write(res["answer"])
        st.caption(f"Latency: {res['latency_ms']:.0f} ms • Model: {res['model']}")
if sum_disabled:
    st.caption("Run a search first to enable  Analyst Report and  Summary.")

# ---------------- render ----------------
# The results are shown in the main area of the app, listed under "Unstructured Sources."
# each search result is displayed in an expandable component (st.expander), with the top result expanded by default.
# the header for each result includes its rank, a hybrid score, and an MMR rank (if applicable).
# the body of the expander shows the relevant text from the document and its metadata.
# ---------------- 

st.subheader("Unstructured Sources")
if not hits:
    st.caption("Run a search to see results.")
else:
    for i, h in enumerate(hits, 1):
        meta = h.get("meta", {})
        hyb = meta.get("hybrid_score", h.get("score", 0.0))
        mmr_rank = meta.get("mmr_rank")
        badges = []
        if mmr_rank:
            badges.append(f"mmr_rank={mmr_rank}")
        badges.append(f"hybrid={hyb:.3f}")
        badges.append(f"alpha={meta.get('alpha', alpha):.2f}")
        src = meta.get("source", "hybrid")
        header = f"**[U{i}]** • {', '.join(badges)} • source: `{src}`"
        with st.expander(header, expanded=(i == 1)):
            st.write(h.get("text", "").strip())
            st.caption(str(meta))

# ---- Insight (auto-extract params from query) ----
# note: very limited function here, just for the demo 
st.divider()
st.subheader("Insight")

# 1) no show if no query yet (and at the start)
if llm_disabled or not query or not query.strip():
    st.info("not avaiable.")
else:
    # Extract ticker (required for insight)
    m_tkr = re.search(r"\b(AAPL|MSFT|NVDA)\b", query, flags=re.I)
    ticker = m_tkr.group(1).upper() if m_tkr else None

    # 2) no show if no ticker in query
    if not ticker:
        st.info("not avaiable.")
    else:
        # Year: default to 2024 if not provided, since only 2024 data available 
        m_year = re.search(r"\b(20\d{2})\b", query)
        year = int(m_year.group(1)) if m_year else 2024

        # Quarters: detect from query; if one found, include the previous quarter too.
        qs = re.findall(r"\bQ([1-4])\b", query.upper())  # capture digits 1–4
        if qs:
            qnums = sorted(set(int(q) for q in qs), reverse=True)
            # if user gave only one quarter, add previous one if possible
            if len(qnums) == 1 and qnums[0] > 1:
                qnums.append(qnums[0] - 1)
            # convert back to Q1..Q4 labels in descending order
            quarters = [f"Q{q}" for q in sorted(qnums, reverse=True)]
        else:
            quarters = ["Q2", "Q1"]  # default pair

        # Build bundle with extracted params
        bundle = combine_sources(query, ticker, year, quarters)        
        fin = (bundle.get("financials") or {}).get("data", [])
        ins = compute_insight(fin)
        if ins:
            label, val, cite = ins
            st.success(f"{label}: {val} [{cite}]")
        else:
            st.info("not avaiable.")

# JSON artifact (for debugging / exports)
if show_json and hits:
    st.subheader("Run JSON")
    st.json({"query": query, "k": k, "alpha": alpha, "lam": lam,
             "filters": {"hide_headers": hide_headers, "scope_aapl_2024": scope_toggle},
             "hits": hits})

# export latest run (if `res` from llm_answer or `hits` from search)
if show_json and hits:
    st.download_button(
        "Download results JSON",
        data=json.dumps({"query":query, "hits":hits}, indent=2),
        file_name="finrag_results.json",
        mime="application/json"
    )