import json
import os
import random
import re
import sys
import time

import pandas as pd

# make mock_llm.py (in the repo root) importable from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm
from mock_llm import est_tokens

os.makedirs("outputs", exist_ok=True)

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATS = ["billing", "delivery", "technical", "fraud"]
OK_STATUSES = ["classified_json", "classified_keyword"]

# ---------------- redaction (order matters) ----------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")
REDACTIONS = [("EMAIL", EMAIL_RE, "[EMAIL]"),
              ("PHONE", PHONE_RE, "[PHONE]"),
              ("ORDER_ID", ORDER_RE, "[ORDER_ID]")]


def redact(text, counts=None):
    for name, pat, placeholder in REDACTIONS:
        text, n = pat.subn(placeholder, text)
        if counts is not None:
            counts[name] += n
    return text


# ---------------- prompts ----------------
ZERO_SHOT = ('You are a support ticket router. CLASSIFY the customer ticket into exactly one '
             'category: billing, delivery, technical or fraud. Reply with JSON only, in the '
             'form {"category": "...", "reason": "..."}.')

FEW_SHOT = (ZERO_SHOT + "\n"
            "EXAMPLES:\n"
            'Ticket: I was charged twice for order [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: My order [ORDER_ID] has not arrived and the rider is not answering. -> {"category": "delivery", "reason": "missing delivery"}\n'
            'Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}\n'
            'Ticket: Someone used my wallet without my permission, this is fraud. -> {"category": "fraud", "reason": "unauthorized use"}')

assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT


# ---------------- safe, resilient model call ----------------
def assert_no_pii(messages):
    for m in messages:
        for name, pat, _ in REDACTIONS:
            assert not pat.search(m["content"]), f"{name} found in a prompt"


def call_with_retry(messages, ticket_id, stats, label, retry_log):
    assert_no_pii(messages)
    n = 0
    while True:
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            stats["transient"] += 1
            if n >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            assert BASE_DELAY * 2 ** n * 0.9 <= wait <= BASE_DELAY * 2 ** n * 1.15
            retry_log.append(f"[{label}] ticket={ticket_id} retry={n} wait={wait:.4f}s")
            time.sleep(wait)
            n += 1


# ---------------- output validation ----------------
def validate(text):
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (text or "").lower()))
        found = words & set(CATS)
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", None
    if isinstance(data, dict) and data.get("category") in CATS:
        return "classified_json", data["category"]
    return "flagged_invalid_output", None


# ---------------- one full pipeline run ----------------
def run_pipeline(tickets, system_prompt, label, token_budget=None, retry_log=None):
    mock_llm.reset()
    stats = {"transient": 0, "permanent": 0}
    total_tokens = 0
    rows = []
    for t in tickets:
        text = t["text"]
        row = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"],
               "true_category": t["true_category"], "status": None,
               "category": None, "tokens_used": 0}
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": redact(text)}]
            try:
                resp = call_with_retry(messages, t["ticket_id"], stats, label, retry_log)
            except mock_llm.TransientLLMError:
                stats["permanent"] += 1
                row["status"] = "llm_error"
            else:
                used = resp.usage.total_tokens
                total_tokens += used
                row["tokens_used"] = used
                row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), total_tokens, stats


def summarize(df, total_tokens, stats, label):
    ok = df["status"].isin(OK_STATUSES)
    n = int(ok.sum())
    correct = int((df.loc[ok, "category"] == df.loc[ok, "true_category"]).sum())
    print(f"\n=== {label} ===")
    print(df["status"].value_counts().to_string())
    print(f"classified: {n} | correct: {correct} | accuracy: {correct / n * 100:.1f}%")
    print(f"total tokens: {total_tokens} | tokens per classified ticket: {total_tokens / n:.1f}")
    print(f"TransientLLMErrors: {stats['transient']} | tickets with a permanent error: {stats['permanent']}")
    return ok


# ======================= MAIN =======================
random.seed(42)
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str).to_dict("records")
print("tickets:", len(tickets))

# Task 1: guardrails
empty = [t for t in tickets if not t["text"].strip()]
long_ = [t for t in tickets if t["text"].strip() and est_tokens(t["text"]) > MAX_INPUT_TOKENS]
print("empty tickets:", len(empty), "| long tickets:", len(long_),
      "| est_tokens of a long ticket:", est_tokens(long_[0]["text"]))

# Task 2: redaction counts over all 164 tickets
counts = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
redacted = [redact(t["text"], counts) for t in tickets]
print("\nsubstitutions:", counts, "| total:", sum(counts.values()), "(expected 55, 91, 132, 278)")
left = sum(len(p.findall(r)) for r in redacted for _, p, _ in REDACTIONS)
print("PII patterns left after redaction:", left, "(expected 0)")

# Task 3: prompt sizes
print("\nzero-shot prompt est tokens:", est_tokens(ZERO_SHOT),
      "| few-shot prompt est tokens:", est_tokens(FEW_SHOT))

# Tasks 4-6: full runs, no budget
retry_log = []
zs, zs_tok, zs_stats = run_pipeline(tickets, ZERO_SHOT, "zero-shot", None, retry_log)
fs, fs_tok, fs_stats = run_pipeline(tickets, FEW_SHOT, "few-shot", None, retry_log)
zs_ok = summarize(zs, zs_tok, zs_stats, "ZERO-SHOT (no budget)")
summarize(fs, fs_tok, fs_stats, "FEW-SHOT (no budget)")
print("\nZero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(zs.loc[zs_ok, "true_category"], zs.loc[zs_ok, "category"]).to_string())

# Task 7a: budget demo
BUDGET_DEMO = 6000
demo, demo_tok, demo_stats = run_pipeline(tickets, ZERO_SHOT, "budget-demo", BUDGET_DEMO, retry_log)
demo[["ticket_id", "customer_id", "status", "category", "tokens_used"]].to_csv(
    "outputs/triage_budget_demo.csv", index=False)
print(f"\n=== BUDGET DEMO (zero-shot, TOKEN_BUDGET={BUDGET_DEMO}) ===")
print(demo["status"].value_counts().to_string())
print("budget_exhausted tickets:", int((demo["status"] == "budget_exhausted").sum()),
      "| tokens used:", demo_tok)

# Task 7b: final run with routing
final, fin_tok, fin_stats = run_pipeline(tickets, FEW_SHOT, "final", 100000, retry_log)
seg = pd.read_csv("outputs/customer_segments.csv")
seg_map = seg.set_index("customer_id")[["segment_name", "is_premium"]].to_dict("index")

routes, priorities, seg_names = [], [], []
for r in final.itertuples():
    info = seg_map.get(r.customer_id)
    seg_names.append(info["segment_name"] if info else None)
    premium = bool(info["is_premium"]) if info else False
    if r.status not in OK_STATUSES:
        route, prio = "human_review", "NORMAL"
    elif r.category == "fraud":
        route, prio = "security_team", "HIGH"
    elif premium:
        route, prio = "premium_desk", "HIGH"
    else:
        route, prio = "support_queue", "NORMAL"
    routes.append(route)
    priorities.append(prio)
final["segment_name"] = seg_names
final["priority"] = priorities
final["route"] = routes
final[["ticket_id", "customer_id", "status", "category", "segment_name",
       "priority", "route", "tokens_used"]].to_csv("outputs/triage_results.csv", index=False)
print("\n=== FINAL (few-shot, TOKEN_BUDGET=100000) ===")
print("rows:", len(final), "(expected 164)")
print(final["status"].value_counts().to_string())
print(final["route"].value_counts().to_string())

# retry log evidence
with open("outputs/retry_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(retry_log) + "\n")
waits = [float(l.split("wait=")[1].rstrip("s")) for l in retry_log]
print(f"\nretry log lines: {len(retry_log)} | wait range: {min(waits):.4f}-{max(waits):.4f}s"
      " (retry 0 must lie in 0.045-0.0575)")
print("sample:", retry_log[0])