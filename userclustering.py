
'''
This code is used to perform cluster analysis on the shopping behavior of Instacart users, 
to divide users into several representative groups, such as "loyal users" and "large buyers".
Why is it important? We may design some marketing strategies to make customers buy more goods.
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

orders = pd.read_csv(r"C:\Users\Richard Shen\Desktop\archive\orders.csv")
prior = pd.read_csv(r"C:\Users\Richard Shen\Desktop\archive\order_products__prior.csv")
train_labels = pd.read_csv(r"C:\Users\Richard Shen\Desktop\archive\order_products__train.csv")
products = pd.read_csv(r"C:\Users\Richard Shen\Desktop\archive\products.csv")

orders["order_id"] = orders["order_id"].astype(int)
prior["order_id"] = prior["order_id"].astype(int)
#I merge the prior data with orders to get information about all items purchased by each user in each order.
prior = prior.drop(columns=["user_id", "order_number"], errors="ignore")
prior = prior.merge(orders[["order_id", "user_id", "order_number"]], on="order_id", how="left")

user_cluster_features = prior.groupby("user_id").agg(
    total_orders=("order_id", "nunique"),
    total_products=("product_id", "count"),
    avg_products_per_order=("product_id", lambda x: x.count() / x.nunique()),
).reset_index()
'''
In order to avoid the influence of feature value differences on clustering, 
I use standardization to convert each feature into data with a mean of 0 and a standard deviation of 1.
'''
scaler = StandardScaler()
user_scaled = scaler.fit_transform(user_cluster_features.drop(columns="user_id"))
'''
Use KMeans and set number of clusters=3 to divide users into 3 groups. 
The algorithm groups users according to their purchasing behavior characteristics.
'''
kmeans = KMeans(n_clusters=3, random_state=42)
user_cluster_features["cluster"] = kmeans.fit_predict(user_scaled)
#I set purchasing behavior characteristics
def label_cluster(row):
    if row["total_orders"] > 50:
        return "Loyal Customer"
    elif row["avg_products_per_order"] > 6:
        return "Bulk Buyer"
    else:
        return "Normal Customer"
#Label each user with a tag
user_cluster_features["cluster_label"] = user_cluster_features.apply(label_cluster, axis=1)
'''
Output the top 20 sample users in each category by grouping by label, 
showing a representative sample of each group.
'''
for label in user_cluster_features["cluster_label"].unique():
    print(f"\nSample users in group: {label}")
    print(user_cluster_features[user_cluster_features["cluster_label"] == label]
      .head(20)[["user_id", "total_orders", "avg_products_per_order"]]
      .to_string(index=False))
#Calculate the average index for each cluster group
cluster_profiles = user_cluster_features.groupby("cluster").agg({
    "total_orders": "mean",
    "total_products": "mean",
    "avg_products_per_order": "mean"
}).round(2)

print("\nCluster behavioral profiles (averages):")
print(cluster_profiles)

label_map = user_cluster_features.groupby("cluster")["cluster_label"].first()
cluster_profiles["label"] = cluster_profiles.index.map(label_map)

metrics = ["total_orders", "total_products", "avg_products_per_order"]

plt.figure(figsize=(8, 5))
ax = sns.barplot(x="label", y="total_orders", data=cluster_profiles, palette="Set2", errorbar=None)
plt.title("Average Total Orders by User Group")
plt.xlabel("User Group")
plt.ylabel("Total Orders")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", label_type="edge")

plt.tight_layout()
plt.show()