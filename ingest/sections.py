# ingest/sections.py
import re

# section list, from specific (risk factors) to general (buinses)
SECTION_PATTERNS = [
    (r"\bItem\s+1A\.?\s+Risk Factors\b", "Risk Factors"),
    (r"\bItem\s+7\.?\s+.*Management’s Discussion", "MD&A"),
    (r"\bItem\s+7A\.?\s+.*Quantitative.*Market Risk", "Market Risk"),
    (r"\bItem\s+1\.?\s+Business\b", "Business"),
]

def detect_section(text: str) -> str:
    """
    the helper function stops and returns as soon as it finds a match, the items higher up in the list 
    (like "Risk Factors") have a higher priority and will be checked first. 
    If there were a text that contained both "Risk Factors" and "Business," 
    the function would return "Risk Factors" and stop
    """  
    t = " ".join(text.split())
    for pat, name in SECTION_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return name
    return "General"
