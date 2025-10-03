import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# Paths
data_path = os.path.join("data", "customers.csv")
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)
print(df.head())

# Features
features = ["AnnualIncome_k$", "SpendingScore", "Age"]
X = df[features].values

# Standardize
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(Xs)

plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], s=40, alpha=0.8)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA (2D) of Features")
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "pca_2d.png"))
plt.close()

# Elbow method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Xs)
    inertias.append(km.inertia_)

plt.figure(figsize=(6,5))
plt.plot(range(1,11), inertias, marker="o")
plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("Elbow Method For Optimal k")
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "elbow_inertia.png"))
plt.close()

# KMeans clustering (k=4 chosen)
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km.fit_predict(Xs)
df["Cluster"] = labels

# Cluster visualization (PCA)
plt.figure(figsize=(6,5))
for cl in sorted(df["Cluster"].unique()):
    idx = df["Cluster"] == cl
    plt.scatter(X_pca[idx,0], X_pca[idx,1], label=f"Cluster {cl}", s=50, alpha=0.8)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Clusters (PCA space)")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(plots_dir, "clusters_pca.png"))
plt.close()

# Scatter with clusters in original space
plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x="AnnualIncome_k$", y="SpendingScore", hue="Cluster", palette="tab10")
plt.title("Income vs Spending Score (colored by cluster)")
plt.savefig(os.path.join(plots_dir, "income_spending_clusters.png"))
plt.close()

# Save clustered dataset
df.to_csv(os.path.join("data", "customers_clustered.csv"), index=False)

# Print evaluation
score = silhouette_score(Xs, labels)
print(f"Silhouette Score (k={k}):", round(score, 4))
print("Cluster sizes:\n", df["Cluster"].value_counts())
