import re
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()

parts = re.split(r"(?m)^-- (Q\d)[^\n]*\n", text)
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}

res = {name: pd.read_sql_query(sql, conn) for name, sql in queries.items()}

print("Q1 rows:", len(res["Q1"]), "(expected 30)")

q2 = res["Q2"]
print("Q2 rows:", len(q2), "(expected 30)")
print("Q2 total revenue:", round(q2["revenue"].sum(), 2), "(expected 2998668.8)")
print("Q2 largest cell:", q2.iloc[0].to_dict(), "(expected Bengaluru / Electronics 233909.45)")

print("Q3 rows:", len(res["Q3"]), "(expected 11)")
print("Q4 rows:", len(res["Q4"]), "(expected 25)")

q5 = res["Q5"]
print("Q5 rows:", len(q5), "(expected 550)")
print("Q5 average:", q5["avg_total"].iloc[0], "(expected 2552.06)")

q6 = res["Q6"]
print("\nQ6 top customer per city (expected C0082, C0930, C0291, C0743, C0693, C0426):")
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:",
      int((q6["rank_pos"] != q6["row_num"]).sum()), "(expected 8)")

q7 = res["Q7"]
print("\nQ7 rows:", len(q7), "(expected 183)")
print("Q7 last cumulative revenue:", q7["cumulative_revenue"].iloc[-1], "(expected 2998668.8)")

conn.close()