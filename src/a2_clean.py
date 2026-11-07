import os
import sqlite3
import pandas as pd

RAW = "data/raw/"
CLEAN = "data/clean/"
os.makedirs(CLEAN, exist_ok=True)

customers = pd.read_csv(RAW + "customers.csv")
products = pd.read_csv(RAW + "products.csv")
orders = pd.read_csv(RAW + "orders.csv")

# ---------- CUSTOMERS ----------
customers["city"] = customers["city"].str.strip().str.title()

g = customers["gender"].fillna("").str.strip().str.lower()
customers["gender"] = g.map({"male": "male", "m": "male",
                             "female": "female", "f": "female"}).fillna("unknown")

bad_age = ~customers["age"].between(10, 100) & customers["age"].notna()
print("ages set to missing (outside 10-100):", int(bad_age.sum()))
customers["age"] = customers["age"].where(customers["age"].between(10, 100))
n_missing_age = int(customers["age"].isna().sum())
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median"))
print("ages filled with city median:", n_missing_age)

customers["signup_date"] = pd.to_datetime(customers["signup_date"])

print("\nCities:\n", customers["city"].value_counts())
print("Genders:\n", customers["gender"].value_counts())
print("Missing ages:", int(customers["age"].isna().sum()),
      "| min/max age:", customers["age"].min(), customers["age"].max())

# ---------- ORDERS ----------
# Rule 1: datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Rule 2: exact duplicates
before = len(orders)
orders = orders.drop_duplicates()
print("\nRule 2 - exact duplicates removed:", before - len(orders))

# Rule 3: keep latest record per order_id
before = len(orders)
orders = (orders.sort_values("order_ts")
                .drop_duplicates("order_id", keep="last")
                .reset_index(drop=True))
print("Rule 3 - older duplicate records removed:", before - len(orders))

# Rule 4: amount to numeric, fill from product price
orders = orders.merge(products[["product_id", "category", "price_inr"]],
                      on="product_id", how="left")
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
n_amt = int(orders["order_amount"].isna().sum())
calc = (orders["price_inr"] * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(calc)
print("Rule 4 - amounts filled from product price:", n_amt)

# Rule 5: delivery minutes
neg = orders["delivery_minutes"] < 0
print("Rule 5 - negative delivery_minutes set to missing:", int(neg.sum()))
orders.loc[neg, "delivery_minutes"] = pd.NA
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"])

comp = orders["status"] == "completed"
cat_median = (orders.loc[comp]
              .groupby("category")["delivery_minutes"].transform("median"))
n_fill = int((comp & orders["delivery_minutes"].isna()).sum())
orders.loc[comp, "delivery_minutes"] = orders.loc[comp, "delivery_minutes"].fillna(cat_median)
print("Rule 5 - completed delivery times filled with category median:", n_fill)

orders = orders.drop(columns=["category", "price_inr"])

# ---------- CHECKS ----------
comp = orders["status"] == "completed"
print("\nOrders rows:", len(orders))
print(orders["status"].value_counts())
print("Missing order_amount:", int(orders["order_amount"].isna().sum()))
print("Missing delivery (completed):", int(orders.loc[comp, "delivery_minutes"].isna().sum()))
print("Missing delivery (cancelled):", int(orders.loc[~comp, "delivery_minutes"].isna().sum()))
print("Completed revenue (INR):", round(orders.loc[comp, "order_amount"].sum(), 2))

# ---------- SAVE ----------
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")

customers.to_csv(CLEAN + "customers.csv", index=False)
products.to_csv(CLEAN + "products.csv", index=False)
orders.to_csv(CLEAN + "orders.csv", index=False)

conn = sqlite3.connect("data/instamart.db")
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()
print("\nSaved clean CSVs and data/instamart.db")