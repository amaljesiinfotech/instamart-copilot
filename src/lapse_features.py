import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")
NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
       "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]


def build_cohort():
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv("data/clean/products.csv")
    orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

    # FEATURES: only orders dated before the cutoff
    before = orders[orders["order_ts"] < CUTOFF]
    comp = before[before["status"] == "completed"]

    feat = (comp.groupby("customer_id")
                .agg(last_order=("order_ts", "max"),
                     frequency=("order_id", "nunique"),
                     monetary=("order_amount", "sum"),
                     avg_discount=("discount_pct", "mean"),
                     avg_delivery=("delivery_minutes", "mean"))
                .reset_index())
    feat["recency"] = (CUTOFF - feat["last_order"]).dt.days

    fav = (comp.groupby("customer_id")["category"]
               .agg(lambda s: s.value_counts().idxmax())
               .rename("fav_category").reset_index())
    cancel = (before.groupby("customer_id")["status"]
                    .apply(lambda s: (s == "cancelled").mean())
                    .rename("cancel_rate").reset_index())

    df = (feat.merge(fav, on="customer_id")
              .merge(cancel, on="customer_id")
              .merge(customers[["customer_id", "city", "age", "gender", "signup_date"]],
                     on="customer_id"))
    df["tenure_days"] = (CUTOFF - df["signup_date"]).dt.days

    # LABEL: no completed order from 2025-09-01 to 2025-09-30
    sept = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END)
                  & (orders["status"] == "completed")]
    df["lapsed"] = (~df["customer_id"].isin(set(sept["customer_id"]))).astype(int)
    return df


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])


def get_split():
    df = build_cohort()
    X, y = df[NUM + CAT], df["lapsed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    return df, X_train, X_test, y_train, y_test