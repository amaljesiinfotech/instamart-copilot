import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = orders[orders["status"] == "completed"]
snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = (comp.groupby("customer_id")
           .agg(last_order=("order_ts", "max"),
                frequency=("order_id", "nunique"),
                monetary=("order_amount", "sum"))
           .reset_index())
rfm["recency"] = (snapshot - rfm["last_order"]).dt.days
rfm["monetary"] = rfm["monetary"].round(2)
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
rfm.to_csv("outputs/rfm.csv", index=False)

print("customers:", len(rfm), "(expected 1175)")
print("recency  median:", rfm["recency"].median(), "min:", rfm["recency"].min(),
      "max:", rfm["recency"].max(), "(expected 19, 0, 180)")
print("frequency mean:", round(rfm["frequency"].mean(), 2),
      "max:", rfm["frequency"].max(), "(expected 9.46, 33)")
print("monetary  mean:", round(rfm["monetary"].mean(), 2),
      "max:", rfm["monetary"].max(), "(expected 2552.06, 14766.55)")