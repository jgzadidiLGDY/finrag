"""This code is an evaluation framework designed to process retrieval results ("artifacts"), 
compares them against a set of expected correct results ("gold standard"), and 
calculates a performance metric (Precision@k), and generate a summary of the results
"""
import json, glob, os
from pathlib import Path
from statistics import mean

GOLD_PATH = "eval/gold.jsonl"  # "gold standard" file
ARTIFACT_GLOB = "data/artifacts/run_*.json"  # retrieval run files

# reads a "gold standard" file
def load_gold(path=GOLD_PATH):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def precision_at_k_loose(hit_ids, expected_ids):
    """
    it calculates the "Precision@k" metric for a single run, 
      > compares the list of IDs returned by a search against the list of expected correct IDs.
      > does substring match, is case-insensitive, and safely handles ints/None.
      > the result is the fraction of the returned hits that matched at least one expected ID
    """
    if not hit_ids:
        return 0.0
    exp = [str(e).lower() for e in expected_ids if e is not None]
    if not exp:
        return 0.0
    cnt = 0
    for hid in hit_ids:
        hid_s = str(hid).lower() if hid is not None else ""
        if any(e in hid_s for e in exp):
            cnt += 1
    return cnt / len(hit_ids)

#  loads multiple run result files
def collect_runs(pattern=ARTIFACT_GLOB):
    for p in glob.glob(pattern):
        with open(p, "r", encoding="utf-8") as f:
            yield json.load(f)

def infer_mode(bundle):
    """Best-effort mode label for grouping."""
    # Prefer explicit
    m = bundle.get("mode")
    if m: return m
    # Heuristics
    lam = bundle.get("lam")
    alpha = bundle.get("alpha")
    if lam is not None and lam not in (0, 0.0):
        return "hybrid+MMR"
    if alpha is not None:
        return "hybrid"
    return "hybrid_or_mmr"

def main():
    """
    it iterates through every artifact file and every query within the gold standard file. 
      > For each query/run combination, it determines the returned IDs, finds the corresponding expected IDs, 
        and calculates the Precision@k score using the precision_at_k_loose function.
    it also collects other metadata like latency, configuration parameters.
    """
    gold_list = load_gold()
    gold = {g["query"]: set(g["expected_ids"]) for g in gold_list}

    rows = []
    for run in collect_runs():
        q = run.get("query", "")
        bundle = run.get("bundle", {})
        hits = bundle.get("faiss_hits", []) or []
        # Try multiple places for ID
        hit_ids = []
        for h in hits:
            hid = h.get("id")
            if not hid:
                meta = h.get("meta", {}) or {}
                hid = meta.get("id") or meta.get("source_id") or meta.get("source")
            hit_ids.append(hid)

        expected = gold.get(q, set())
        p_at_k = precision_at_k_loose(hit_ids, expected)
        rows.append({
            "query": q,
            "k": len(hits),
            "precision@k": round(p_at_k, 3),
            "latency_ms": round(run.get("latency_ms", 0.0), 1),
            "alpha": bundle.get("alpha"),
            "lam": bundle.get("lam"),
            "mode": infer_mode(bundle),
        })

    # Write detailed results
    Path("eval").mkdir(parents=True, exist_ok=True)
    Path("eval/results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    #------ generate summary ----- 
    
    # Summarize by mode
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], {"pks": [], "lats": []})
        by_mode[r["mode"]]["pks"].append(r["precision@k"])
        by_mode[r["mode"]]["lats"].append(r["latency_ms"])

    summary = []
    for mode, vals in by_mode.items():
        summary.append({
            "mode": mode,
            "mean_P@k": round(mean(vals["pks"]), 3) if vals["pks"] else 0.0,
            "avg_latency_ms": round(mean(vals["lats"]), 1) if vals["lats"] else 0.0,
            "n_runs": len(vals["pks"]),
        })

    # Sort summary by mean_P@k desc
    summary.sort(key=lambda x: x["mean_P@k"], reverse=True)
    Path("eval/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print compact table
    if summary:
        print("\nmode          mean_P@k    avg_latency_ms    n")
        print("-" * 48)
        for s in summary:
            print(f"{s['mode']:<13} {s['mean_P@k']:<11.3f} {s['avg_latency_ms']:<15.1f} {s['n_runs']}")
        print("\nWrote eval/results.json and eval/summary.json")
    else:
        print("No runs summarized. Make sure artifacts exist in data/artifacts/ and gold.jsonl is set.")

if __name__ == "__main__":
    main()
