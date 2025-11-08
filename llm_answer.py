import time, re, os, json
from openai import OpenAI
from fuse import combine_sources
from render_prompt import render

CITATION_RE = re.compile(r"\[(U|S)\d+\]")

#------- part of prompt building, or Prompt Engineering 
# system rules: specific rules for LLM to achieve desired output format 
SYSTEM_RULES = (
    "You are a cautious financial analyst.\n"
    "RULES:\n"
    "1) Use ONLY the sources listed in the prompt.\n"
    "2) Cite every factual claim using [U#] or [S#] right after the sentence.\n"
    "3) Do NOT invent citations, numbers, or sources.\n"
    "4) If the sources are insufficient, say so explicitly.\n"
    "5) Keep the answer concise (5–8 sentences).\n"
)

# addtional rules:  retry if missing citations 
REPAIR_INSTR = (
    "Your previous answer did not include valid [U#]/[S#] citations. "
    "Revise the answer now. Every factual sentence must include at least one matching citation tag. "
    "Use only tags that appear in the SOURCES section."
)

# to check any citations 
def _has_valid_cites(text: str) -> bool:
    return bool(CITATION_RE.search(text or ""))

# connects to LLM and to get LLM's answer 
def _call(client, prompt):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":SYSTEM_RULES},
            {"role":"user","content":prompt}
        ],
        temperature=0,
        max_tokens=700
    )
    return resp.choices[0].message.content, (time.perf_counter()-t0)*1000

def llm_answer(query, ticker=None, year=None, quarters=None):
    """
    it sends a preparted prompt to the OpenAI API, measures the time it takes to get a response, and 
    returns the prompt, the LLM's answer, and the API latency
    """     
    bundle = combine_sources(query, ticker, year, quarters)
    prompt = render({"query": query, **bundle})  # ensure QUESTION + SOURCES present

    client = OpenAI()

    text, latency_ms = _call(client, prompt)
    # retry if missing citations 
    if not _has_valid_cites(text):
        # one repair attempt with explicit instruction + prior answer context trimmed out
        repair_prompt = prompt + "\n\n=== INSTRUCTION ===\n" + REPAIR_INSTR
        repaired, latency_ms2 = _call(client, repair_prompt)
        if _has_valid_cites(repaired):
            text = repaired
            latency_ms += latency_ms2
        else:
            text += "\n\n*Note: No valid [U#]/[S#] citations detected. Please verify sources.*"

    # preparing to store process data for potential auditing 
    result = {
        "query": query, 
        "ticker": ticker, 
        "year": year, 
        "quarters": quarters,
        "prompt": prompt, 
        "answer": text, 
        "bundle": bundle,
        "latency_ms": latency_ms, 
        "model": "gpt-4o-mini",
        "timestamp": time.strftime("%Y%m%d-%H%M%S")
    }

    os.makedirs("data/artifacts", exist_ok=True)
    path = f"data/artifacts/run_{ticker or 'NA'}_{year or 'NA'}_{result['timestamp']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)  # save to file 
    print(f"Saved: {path}  |  Latency: {latency_ms:.0f} ms")

    return result
