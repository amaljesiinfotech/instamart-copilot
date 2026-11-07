import pandas as pd

RAW = "data/raw/"
customers = pd.read_csv(RAW + "customers.csv")
products = pd.read_csv(RAW + "products.csv")
orders = pd.read_csv(RAW + "orders.csv")

for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print(f"\n=== {name}: shape {df.shape} ===")
    print(df.dtypes)
    miss = pd.DataFrame({"missing": df.isna().sum(),
                         "pct": (df.isna().mean() * 100).round(2)})
    print(miss)

amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": int(amount_num.isna().sum()),
    "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(customers["age"].isna().sum()),
    "age outside 10-100": int(((customers["age"] < 10) | (customers["age"] > 100)).sum()),
    "distinct raw city spellings": int(customers["city"].nunique()),
    "distinct non-blank gender spellings": int(customers["gender"].nunique()),
    "blank gender": int(customers["gender"].isna().sum()),
}
print("\n=== AUDIT FIGURES ===")
for k, v in audit.items():
    print(f"{k}: {v}")