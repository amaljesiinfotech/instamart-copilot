import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rfm = pd.read_csv("outputs/rfm.csv")
cols = ["recency", "frequency", "monetary"]
X = StandardScaler().fit_transform(rfm[cols])

km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

# (a) profile
prof = (rfm.groupby("segment_id")
           .agg(size=("customer_id", "count"),
                mean_recency=("recency", "mean"),
                mean_frequency=("frequency", "mean"),
                mean_monetary=("monetary", "mean"))
           .round(2))
print("=== Segment profile ===")
print(prof)

# (b) names assigned by rule, because cluster numbers are only labels
premium_id = prof["mean_monetary"].idxmax()
rest = prof.drop(premium_id)
loyal_id = rest["mean_frequency"].idxmax()
rest = rest.drop(loyal_id)
risk_id = rest["mean_recency"].idxmax()
other_id = rest.drop(risk_id).index[0]

names = {premium_id: "Premium Big Spenders",
         loyal_id: "Loyal Frequent Buyers",
         risk_id: "At-Risk Customers",
         other_id: "Occasional Low-Value Buyers"}
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id

print("\n=== Segment names ===")
for sid in prof.index:
    print(sid, "->", names[sid], "| premium:", sid == premium_id)

# (c) PCA
pca = PCA(n_components=2).fit(X)
r = pca.explained_variance_ratio_
print("\nPCA explained variance ratio:", r.round(3), "| total:", round(r.sum(), 3))
print("Expected: about 0.634 and 0.218, total about 0.852")

out = rfm[["customer_id", "recency", "frequency", "monetary",
           "segment_id", "segment_name", "is_premium"]]
out.to_csv("outputs/customer_segments.csv", index=False)
print("\nSaved outputs/customer_segments.csv, rows:", len(out))