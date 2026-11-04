"""Read-only local tools for the agent.

Every tool is read-only by design: there is no tool that writes files,
sends messages, changes state, or spends money. A tool failure is returned
as an error value, never papered over with an invented result.
"""
import ast
import csv
import math
import os
import re
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def search_documents(query, top_k=2):
    """Simple keyword search over local markdown docs. Returns snippets."""
    docs_dir = os.path.join(BASE_DIR, "data", "docs")
    stopwords = {
        "what", "is", "the", "of", "a", "an", "for", "how", "when", "why",
        "does", "do", "are", "was", "were", "in", "on", "to", "me", "tell",
    }
    query_words = set(re.findall(r"[a-z0-9]+", query.lower())) - stopwords
    scored = []
    for name in sorted(os.listdir(docs_dir)):
        if not name.endswith(".md"):
            continue
        path = os.path.join(docs_dir, name)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        overlap = len(query_words & words)
        if overlap:
            scored.append((overlap, name, text))
    scored.sort(key=lambda x: -x[0])
    results = []
    for _, name, text in scored[:top_k]:
        # Return the most relevant sentence-ish chunk, capped for the trace.
        flat = " ".join(text.split())
        results.append({"source": name, "snippet": flat[:400]})
    if not results:
        return {"error": f"No local document matched: {query!r}"}
    return {"results": results}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)
    ):
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        return left ** right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculator(expression):
    """Evaluate a numeric expression safely (no eval(), no names, no calls)."""
    expr = str(expression).strip()
    if not expr or not re.fullmatch(r"[0-9\s\.\+\-\*/%\(\)]+", expr):
        return {"error": f"Calculator cannot parse expression: {expression!r}"}
    try:
        tree = ast.parse(expr, mode="eval")
        value = _safe_eval(tree)
    except Exception as exc:  # noqa: BLE001 - surfaced as a tool error
        return {"error": f"Calculator failed on {expression!r}: {exc}"}
    if isinstance(value, float) and math.isfinite(value):
        value = round(value, 4)
    return {"result": value}


_WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100",
}


def normalise_expression(text):
    """Convert a wordy arithmetic phrase into a numeric expression."""
    lowered = text.lower()
    for word, digit in _WORD_NUMBERS.items():
        lowered = re.sub(rf"\b{word}\b", digit, lowered)
    lowered = lowered.replace("times", "*").replace("multiplied by", "*")
    lowered = lowered.replace("plus", "+").replace("minus", "-")
    lowered = lowered.replace("divided by", "/").replace("over", "/")
    # Keep only digits, operators, parentheses, dots and spaces.
    cleaned = re.sub(r"[^0-9\.\+\-\*/%\(\) ]", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Drop a leading/trailing operator left over from the sentence.
    cleaned = re.sub(r"^[\+\*/%]+", "", cleaned).strip()
    cleaned = re.sub(r"[\+\-\*/%]+$", "", cleaned).strip()
    # Collapse accidental double operators (e.g. "12 * * 5" -> "12 * 5").
    cleaned = re.sub(r"([\+\-\*/%])\s*([\+\-\*/%])", r"\1", cleaned)
    return cleaned


def lookup_record(query):
    """Look up one row of the local structured CSV by key or category."""
    path = os.path.join(BASE_DIR, "data", "records.csv")
    needle = query.lower().replace(" ", "-")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["key"] == needle or row["key"] in query.lower().replace(" ", "-"):
            return {"record": row}
    for row in rows:
        if row["category"] in query.lower():
            return {"record": row}
    return {"error": f"No record found for: {query!r}"}


def get_date_time():
    """Return the current UTC date/time. Read-only; used for date questions."""
    now = datetime.now(timezone.utc)
    return {"utc_now": now.isoformat(), "date": now.strftime("%Y-%m-%d")}


TOOLS = {
    "search_documents": search_documents,
    "calculator": calculator,
    "lookup_record": lookup_record,
    "get_date_time": get_date_time,
}
