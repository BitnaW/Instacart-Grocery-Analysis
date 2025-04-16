import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    orders = pd.read_csv("DataSet/orders.csv")
    prior = pd.read_csv("DataSet/order_products__prior.csv")
    train_labels = pd.read_csv("DataSet/order_products__train.csv")
    products = pd.read_csv("DataSet/products.csv")
    return orders, prior, train_labels, products

def create_profile(orders, prior):
    # Add user_id and order_number to prior
    prior = prior.merge(orders[["order_id", "user_id", "order_number"]], on="order_id")

    # USER-PRODUCT INTERACTION FEATURES
    user_product = prior.groupby(["user_id", "product_id"]).agg(
        user_prod_order_count=("order_id", "count"),
        user_prod_avg_cart_order=("add_to_cart_order", "mean"),
    ).reset_index()

    # USER-LEVEL FEATURES
    user_features = prior.groupby("user_id").agg(
        user_total_orders=("order_number", "max"),
    ).reset_index()

    # PRODUCT-LEVEL FEATURES
    product_features = prior.groupby("product_id").agg(
        product_total_purchases=("order_id", "count"),
        product_avg_cart_order=("add_to_cart_order", "mean")
    ).reset_index()

    # MERGE FEATURES INTO SINGLE DF
    features = user_product.merge(user_features, on="user_id", how="left")
    features = features.merge(product_features, on="product_id", how="left")

    return features, user_product

def build_model(orders, train_labels, user_product, features):
    # Get training orders
    train_orders = orders[orders.eval_set == "train"][["user_id", "order_id"]]

    # Get user-product candidates for training
    train_candidates = train_orders.merge(user_product[["user_id", "product_id"]], on="user_id", how="left")

    # Merge with labels
    train_labels = train_labels[["order_id", "product_id", "reordered"]]
    train_set = train_candidates.merge(train_labels, on=["order_id", "product_id"], how="left")
    train_set["reordered"] = train_set["reordered"].fillna(0)

    # Add features
    train_set = train_set.merge(features, on=["user_id", "product_id"], how="left")

    drop_cols = ["user_id", "product_id", "order_id"]
    X = train_set.drop(columns=drop_cols + ["reordered"])
    y = train_set["reordered"]

    return X, y

def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.21).astype(int)

    f1 = f1_score(y_val, y_pred)
    print(f"F1-score: {f1:.4f}")

    return model, X, f1

def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx][::-1])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx][::-1])
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()
    
def recommender(user_id, user_product, user_features, product_features, model, products):
    user_data = user_product[user_product.user_id == user_id]
    user_data = user_data.merge(user_features, on="user_id", how="left")
    user_data = user_data.merge(product_features, on="product_id", how="left")

    X_user = user_data.drop(columns=["user_id", "product_id"])
    print("Model trained on:", list(model.feature_names_in_))
    print("Input columns for prediction:", list(X_user.columns))
    X_user = X_user.reindex(columns=model.feature_names_in_, fill_value=0)
    probs = model.predict_proba(X_user)[:, 1]
    user_data["predicted_reorder_prob"] = probs

    top_products = user_data.sort_values(by="predicted_reorder_prob", ascending=False).head(5)
    return top_products.merge(products, on="product_id")[["product_name", "predicted_reorder_prob"]]

# Optional: This would be in your main file
if __name__ == "__main__":
    orders, prior, train_labels, products = load_data()
    features, user_product = create_profile(orders, prior)
    X, y = build_model(orders, train_labels, user_product, features)
    model, X_full, f1 = train_model(X, y)

    user_id_demo = 1
    recs = recommender(user_id_demo, user_product, features[["user_id", "user_total_orders"]], features[["product_id", "prod_total_purchases", "prod_avg_cart_order"]], model, products)
    print("Top recommendations:\n", recs)
