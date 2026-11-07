import sqlite3
import pandas as pd

customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")
comp = orders["status"] == "completed"
canc = orders["status"] == "cancelled"

# Orders checkpoints
assert len(orders) == 12309
assert orders["order_id"].nunique() == 12309
assert comp.sum() == 11110
assert canc.sum() == 1199
assert not (orders["status"] == "pending").any()
assert orders["order_amount"].isna().sum() == 0
assert orders.loc[comp, "delivery_minutes"].isna().sum() == 0
assert orders.loc[canc, "delivery_minutes"].isna().sum() == 1199
assert round(orders.loc[comp, "order_amount"].sum(), 2) == 2998668.80

# Customers checkpoints
assert customers["city"].value_counts().to_dict() == {
    "Pune": 227, "Delhi": 210, "Bengaluru": 197,
    "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
assert customers["gender"].value_counts().to_dict() == {
    "female": 610, "male": 560, "unknown": 30}
assert customers["age"].isna().sum() == 0
assert customers["age"].between(18, 60).all()

# Database checkpoints
conn = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert {"customers", "products", "orders"} <= tables
assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309
assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 1200
conn.close()

print("ALL CHECKPOINTS PASSED")