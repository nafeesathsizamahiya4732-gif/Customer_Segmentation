import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



df = pd.read_csv("dataset/customer_segmentation.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print("Number of rows and columns:", df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())





df = df.drop(columns=["Number"])

print("\n========== DATASET AFTER PREPROCESSING ==========")
print("Rows and columns:", df.shape)

print("\n========== MISSING VALUES AFTER PREPROCESSING ==========")
print(df.isnull().sum())




X = df[
    [
        "Age",
        "Income",
        "Spending_Score",
        "Membership_Years",
        "Purchase_Frequency",
        "Last_Purchase_Amount"
    ]
]

print("\n========== FEATURES (X) ==========")
print(X.columns)

print("\n========== FEATURE SHAPE ==========")
print(X.shape)




scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n========== FEATURE SCALING COMPLETE ==========")
print("Scaled feature shape:", X_scaled.shape)




inertia = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True)

# Save elbow graph
os.makedirs("images", exist_ok=True)
plt.savefig("images/elbow_method.png")

plt.show()
plt.close()




kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n========== CLUSTERING COMPLETE ==========")
print("Number of clusters:", df["Cluster"].nunique())

print("\n========== CLUSTER COUNTS ==========")
print(df["Cluster"].value_counts().sort_index())




os.makedirs("model", exist_ok=True)

joblib.dump(kmeans, "model/customer_segmentation_model.pkl")

print("\n========== MODEL SAVED ==========")
print("K-Means model saved successfully.")
print("File name: model/customer_segmentation_model.pkl")




cluster_summary = df.groupby("Cluster")[[
    "Age",
    "Income",
    "Spending_Score",
    "Membership_Years",
    "Purchase_Frequency",
    "Last_Purchase_Amount"
]].mean()

print("\n========== CLUSTER SUMMARY ==========")
print(cluster_summary)




plt.figure(figsize=(8, 6))

plt.scatter(
    df["Income"],
    df["Spending_Score"],
    c=df["Cluster"],
    cmap="viridis",
    s=40
)

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.colorbar(label="Cluster")


plt.savefig("images/customer_segments.png")

plt.show()
plt.close()




df.to_csv("customer_segments.csv", index=False)

print("\n========== PROJECT COMPLETE ==========")
print("Clustered customer dataset saved successfully.")
print("File name: customer_segments.csv")