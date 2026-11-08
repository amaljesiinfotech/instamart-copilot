import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm  # noqa: E402

POLICY_DIR = os.path.join(ROOT, "data", "policies")

SYSTEM_PROMPT = ("You are a POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
                 "If the context does not contain the answer, say that you could not find "
                 "the information in the policy documents.")


def chunk_text(text, size=40, overlap=10):
    """Word chunks: first covers words 0-39, next starts 30 words later, and so on."""
    words = text.split()
    step = size - overlap
    chunks, start = [], 0
    while True:
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):          # this chunk reaches or passes the last word
            break
        start += step
    return chunks


def build_chunks(size=40, overlap=10):
    out = []
    for fn in sorted(os.listdir(POLICY_DIR)):
        if fn.endswith(".txt"):
            text = open(os.path.join(POLICY_DIR, fn), encoding="utf-8").read()
            for c in chunk_text(text, size, overlap):
                out.append({"source": fn, "text": c})
    return out


class PolicyIndex:
    def __init__(self, size=40, overlap=10):
        self.chunks = build_chunks(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]),
                 "text": self.chunks[i]["text"]} for i in top]


INDEX = PolicyIndex(40, 10)


def retrieve(query, k=3):
    return INDEX.retrieve(query, k)


def answer(question, index=None):
    index = index or INDEX
    hits = index.retrieve(question, k=3)
    blocks = [f"[{h['source']}]\n{h['text']}" for h in hits]
    user = "CONTEXT:\n" + "\n".join(blocks) + "\nQUESTION:\n" + question
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]


def evaluate(index, label):
    qa = pd.read_csv(os.path.join(ROOT, "data", "raw", "policy_eval.csv"), keep_default_na=False)
    doc_hit = fact_hit = fallback = 0
    n_ans = n_un = 0
    failures = []
    for r in qa.itertuples():
        text, sources = answer(r.question, index)
        if r.expected_doc == "NONE":
            n_un += 1
            if text.startswith("I could not find"):
                fallback += 1
            else:
                failures.append((r.question, "unanswerable question did not get the fallback", text))
        else:
            n_ans += 1
            d_ok = r.expected_doc in sources
            f_ok = r.expected_fact in text
            doc_hit += d_ok
            fact_hit += f_ok
            if not (d_ok and f_ok):
                failures.append((r.question, f"doc_in_top3={d_ok}, fact_in_answer={f_ok}", text))
    print(f"\n=== {label}: {len(index.chunks)} chunks ===")
    print(f"answerable: expected_doc in top 3 = {doc_hit}/{n_ans} | expected_fact in answer = {fact_hit}/{n_ans}")
    print(f"unanswerable: fallback answers = {fallback}/{n_un}")
    for q, why, text in failures:
        print("  FAILS:", q, "|", why, "\n         answer:", text)
    return doc_hit, fact_hit, fallback


if __name__ == "__main__":
    # Task 1: chunk counts
    counts = pd.Series([c["source"] for c in INDEX.chunks]).value_counts()
    print("Chunks per file:\n", counts.to_string())
    print("Total chunks:", len(INDEX.chunks), "(expected 15: refund 4, delivery 3, escalation 3, payment 3, privacy 2)")

    # Task 2: retrieval check
    hits = retrieve("refund policy late delivery")
    print("\nTop 3 for 'refund policy late delivery':")
    for h in hits:
        print(f"  {h['source']}  score={h['score']:.3f}")

    # Task 3: one example answer
    text, sources = answer("Who must approve a refund above INR 1000?")
    print("\nExample answer:", text, "| sources:", sources)

    # Task 4: evaluation with overlap 10 and overlap 0
    evaluate(PolicyIndex(40, 10), "Overlap 10 (40-word chunks)")
    evaluate(PolicyIndex(40, 0), "Overlap 0 (40-word chunks)")