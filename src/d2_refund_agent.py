import glob
import json
import os
import sqlite3
import sys
import time
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import mock_llm  # noqa: E402
from d1_policy_rag import retrieve  # noqa: E402  (retrieval reused from tasks 1-3)

DB_PATH = os.path.join(ROOT, "data", "instamart.db")
OUT_DIR = os.path.join(ROOT, "outputs")
STATE_DIR = os.path.join(ROOT, "state")
LEDGER = os.path.join(OUT_DIR, "refund_ledger.jsonl")
RUNS = os.path.join(OUT_DIR, "agent_runs.jsonl")
DEMOS = os.path.join(OUT_DIR, "agent_demos.jsonl")

AGENT_SYSTEM = ("You are a refund AGENT. Use the tools to look up the order, check the policy, "
                "calculate the refund and then issue it or escalate it to a human. When finished, "
                "reply with a final answer in the form STATUS: explanation, where STATUS is "
                "REFUND, ESCALATE, DENIED or NOT_FOUND.")

_CTX = {"run_id": None}   # lets ledger lines record which run wrote them


class SimulatedCrash(Exception):
    pass


# ---------------------------- tools ----------------------------
def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT order_id, status, order_amount, delivery_minutes "
                       "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    conn.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"),
                            "tool": tool, "run_id": _CTX["run_id"],
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order by id in the Instamart database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the top matching chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund amount in INR from the order amount and delivery time.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                     "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an approved refund and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                     "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                     "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}


# ---------------------------- checkpoints ----------------------------
def _save_checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def _load_checkpoint(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------- the ReAct loop ----------------------------
def _result(state, status, final_answer):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "tools_executed": state["tools_executed"]}


def _fallback(state, status):
    done = ", ".join(state["tools_executed"]) or "none"
    summary = (f"Handoff to human agent ({status}). Request: {state['user_message']} "
               f"Tools completed so far: {done}. Model calls: {state['llm_calls']}, "
               f"tokens used: {state['tokens_used']}.")
    return _result(state, status, summary)


def _loop(state, crash_after_tools=None):
    p = state["params"]
    _CTX["run_id"] = state["run_id"]
    start = time.monotonic()
    while True:
        # (c) wall-clock limit, checked at the top of each iteration
        if time.monotonic() - start >= p["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT")
        # (a) iteration limit: max_iterations model calls without a final answer
        if state["llm_calls"] >= p["max_iterations"]:
            return _fallback(state, "FALLBACK_MAX_ITERATIONS")

        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        # (b) token limit: stop at once, without executing the tool just requested
        if state["tokens_used"] > p["token_limit"]:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT")

        msg = resp.choices[0].message
        if not msg.tool_calls:                      # final answer
            answer = msg.content or ""
            status = answer.split(":")[0].strip() or "UNKNOWN"
            return _result(state, status, answer)

        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name,
                                                  "arguments": tc.arguments}]})
        fn = TOOL_MAP.get(tc.name)
        if fn is None:
            obs = {"error": f"unknown tool {tc.name}"}
        else:
            try:
                obs = fn(**tc.arguments)
            except Exception as e:                  # a bad call becomes an observation, not a crash
                obs = {"error": str(e)}
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(obs)})
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        state["tools_executed"].append(tc.name)
        state["step"] = len(state["steps"])
        _save_checkpoint(state)                     # after every tool observation
        if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or f"run_{scenario_id}", "scenario_id": scenario_id,
             "user_message": user_message,
             "messages": [{"role": "system", "content": AGENT_SYSTEM},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "llm_calls": 0, "steps": [], "tools_executed": [],
             "params": {"max_iterations": max_iterations, "token_limit": token_limit,
                        "time_limit_seconds": time_limit_seconds}}
    return _loop(state, crash_after_tools)


def resume_agent(run_id):
    state = _load_checkpoint(run_id)
    return _loop(state)


# ---------------------------- helpers for the demo ----------------------------
def refund_of(result):
    for s in reversed(result["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"].get("refund_inr", 0)
    return 0


def log_line(path, result):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": result["scenario_id"], "run_id": result["run_id"],
                            "status": result["status"], "final_answer": result["final_answer"],
                            "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"],
                            "steps": result["steps"]}) + "\n")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    for path in (LEDGER, RUNS, DEMOS):             # fresh files so reruns are reproducible
        open(path, "w").close()
    for f in glob.glob(os.path.join(STATE_DIR, "*.json")):
        os.remove(f)

    sc = pd.read_csv(os.path.join(ROOT, "data", "raw", "agent_scenarios.csv"), dtype=str)

    # Task 6: all seven scenarios
    print("=== Seven scenarios ===")
    print(f"{'id':3} {'status':24} {'refund':>9} {'calls':>5} {'tokens':>6}  expected")
    results = {}
    for r in sc.itertuples():
        res = run_agent(r.message, r.scenario_id, run_id=f"{r.scenario_id}_main")
        results[r.scenario_id] = res
        log_line(RUNS, res)
        refund = refund_of(res)
        exp_amt = float(r.expected_refund_inr)
        ok = (res["status"] == r.expected_outcome or
              (r.expected_outcome == "FALLBACK" and res["status"].startswith("FALLBACK"))) \
            and abs(refund - exp_amt) < 0.01
        print(f"{r.scenario_id:3} {res['status']:24} {refund:9.2f} {res['llm_calls']:5} "
              f"{res['tokens_used']:6}  {r.expected_outcome} {exp_amt:.2f} -> {'OK' if ok else 'MISMATCH'}")
    s7 = results["S7"]
    print("S7 uses all six model calls:", s7["llm_calls"] == 6, "| tokens:", s7["tokens_used"],
          "| status:", s7["status"])

    s1 = sc[sc.scenario_id == "S1"].iloc[0]

    # Task 7: stop-safely demonstrations
    print("\n=== Stop-safely demos ===")
    d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1_tokenlimit")
    log_line(DEMOS, d1)
    print("S1 with token_limit=200 ->", d1["status"], "| calls:", d1["llm_calls"],
          "| tokens:", d1["tokens_used"], "| tools executed:", d1["tools_executed"])
    print("   handoff:", d1["final_answer"])
    d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_timeout")
    log_line(DEMOS, d2)
    print("S1 with time_limit_seconds=0 ->", d2["status"], "| calls:", d2["llm_calls"])
    print("   handoff:", d2["final_answer"])

    # Task 8: crash and resume
    print("\n=== Crash and resume (S1) ===")
    try:
        run_agent(s1.message, "S1", run_id="S1_crash", crash_after_tools=2)
    except SimulatedCrash as e:
        print("crashed as planned:", e)
    cp = _load_checkpoint("S1_crash")
    print("checkpoint tools executed before the crash:", cp["tools_executed"])
    res = resume_agent("S1_crash")
    log_line(DEMOS, res)
    print("resumed ->", res["status"], "| refund:", refund_of(res), "| calls:", res["llm_calls"])
    print("tools executed across crash + resume:", res["tools_executed"])
    n_issue = sum(1 for l in open(LEDGER) if json.loads(l)["run_id"] == "S1_crash")
    print("ledger lines written by the crash run:", n_issue, "(must be exactly 1)")
    assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(refund_of(res) - 128.25) < 0.01 and n_issue == 1
    print("state/ files:", sorted(os.listdir(STATE_DIR)))