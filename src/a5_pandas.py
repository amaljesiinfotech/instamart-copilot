import numpy as np
import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

comp = orders[orders["status"] == "completed"].copy()
df = (comp.merge(customers[["customer_id", "city"]], on="customer_id")
          .merge(products[["product_id", "category"]], on="product_id"))

# ---------- (a) pivot table cross-check ----------
pivot = pd.pivot_table(df, values="order_amount", index="city",
                       columns="category", aggfunc="sum",
                       margins=True, margins_name="Total")
print("=== Pivot (revenue by city x category, INR) ===")
print(pivot.round(2))
print("\nPivot grand total:", round(pivot.loc["Total", "Total"], 2),
      "(SQL total: 2998668.8)")

city_stats = (df.groupby("city")
                .agg(total=("order_amount", "sum"),
                     mean=("order_amount", "mean"),
                     count=("order_amount", "count"))
                .round(2))
print("\n=== Per-city sum, mean, count (completed orders) ===")
print(city_stats)
print("Expected: Bengaluru 523643.40 / 276.47 / 1894; Mumbai 479058.50 / 297.00 / 1613")

# ---------- (b) exploratory statistics ----------
x = comp["order_amount"]
q1, q2, q3 = x.quantile(0.25), x.quantile(0.50), x.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

print("\n=== order_amount statistics (completed orders) ===")
print("mean:", round(x.mean(), 2), "| median:", round(x.median(), 2))
print("skewness:", round(x.skew(), 2))
print("Q1:", round(q1, 2), "| Q2:", round(q2, 2), "| Q3:", round(q3, 2), "| IQR:", round(iqr, 2))
print("lower fence:", round(lower, 2), "| outliers below:", int((x < lower).sum()))
print("upper fence:", round(upper, 2), "| outliers above:", int((x > upper).sum()))
print("Expected: mean 269.91, median 135.00, skew ~4.61, Q1 64.80, Q3 297.00, "
      "upper fence 645.30, 897 above, 0 below")

cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = comp[cols].corr()
print("\n=== Correlation matrix ===")
print(corr.round(3))

pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
top = pairs.abs().idxmax()
print("\nLargest |correlation|:", top, "=", round(pairs[top], 3))
print("Expected: order_amount & discount_pct, about -0.095")