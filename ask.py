from retrieval.search import search
from openai import OpenAI
import sys, textwrap

def ask(q, k=5, filters=None):
    """
    The function takes a query and optional arguments to search and display results in a formatted, human-readable way. 
      > the search() function performs the actual lookup.
    """       
    hits = search(q, k=k, filters=filters)
    for i, row in hits.reset_index(drop=True).iterrows():
        header = f"[{i+1}] score={row['score']:.3f} {row['ticker']} {row['doc_type']} {row['section']} p.{int(row['page'])}"
        body = textwrap.shorten(" ".join(row['text'].split()), width=400, placeholder=" …")
        print(header); print(body); print("-"*80)

def answer(query):
    """
    This function uses RAG to answer user's query 
      (1) Retrieve the most relevant data through indexed external (real-time or updated) financial data,
      (2) use this retrieved data as context to Augment user's prompt, 
      (3) ask LLM to anser user's query according to the augmented context/Prompt. 
    """    
    ctx = search(query, k=5) # retrieval 
    context_text = "\n\n---\n\n".join(
        [f"(p.{c['page']} {c['ticker']} {c['doc_type']})\n{c['text']}" for c in ctx]
    )
    system = "You are a cautious financial analyst. Use ONLY the provided context. Cite pages."
    # augment uer's prompt, specifically, the contxt 
    prompt = f"Question: {query}\n\nContext:\n{context_text}\n\nAnswer in 5-7 sentences with citations like (AAPL 10-K p.35)."
    client = OpenAI()
    # asks LLM to generate anser to user's query according to the retrieved data 
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        temperature=0.3  # give LLM a bit flexibility 
    )
    return resp.choices[0].message.content

#if __name__ == "__main__":
#    print(answer("What are the key risks mentioned in Apple's annaul financial report?"))
