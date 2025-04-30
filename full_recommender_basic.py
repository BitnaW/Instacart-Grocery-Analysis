import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from mlxtend.frequent_patterns import fpgrowth, association_rules
import numpy as np
import matplotlib.pyplot as plt
from load_csv import *
from create_profile import *

def build_model(orders, train_labels, user_product, features):
    train_orders = orders[orders.eval_set == "train"][["user_id", "order_id"]]

    train_candidates = train_orders.merge(user_product[["user_id", "product_id"]], on="user_id", how="left")

    train_labels = train_labels[["order_id", "product_id", "reordered"]]
    train_set = train_candidates.merge(train_labels, on=["order_id", "product_id"], how="left")
    train_set["reordered"] = train_set["reordered"].fillna(0)

    train_set = train_set.merge(features, on=["user_id", "product_id"], how="left")

    drop_cols = ["user_id", "product_id", "order_id"]
    X = train_set.drop(columns=drop_cols + ["reordered"])
    y = train_set["reordered"]

    return X, y


def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.21).astype(int)

    f1 = f1_score(y_val, y_pred)
    print(f"F1-score: {f1:.4f}")

    return model, X, f1

def recommender(user_id, user_product, user_features, product_features, model, products):
    user_data = user_product[user_product.user_id == user_id]

    user_data = user_data.merge(user_features, on="user_id", how="left")
    user_data = user_data.merge(product_features, on="product_id", how="left")

    user_data = user_data.groupby(["user_id", "product_id"], as_index=False).agg({
        'user_prod_order_count_clean': 'max',
        'user_prod_avg_cart_order_clean': 'mean',
        'user_total_orders_clean': 'max',
        'product_total_purchases_clean': 'sum',
        'product_avg_cart_order_clean': 'mean',

    })

    rec_user = user_data.drop(columns=["user_id", "product_id"])

    print("trained on: ", list(model.feature_names_in_))
    rec_user = rec_user.reindex(columns=model.feature_names_in_, fill_value=0)

    calc_probs = model.predict_proba(rec_user)[:, 1]
    user_data["predicted_reorder_prob"] = calc_probs

    top_products = user_data.sort_values(by="predicted_reorder_prob", ascending=False).head(11)
    return top_products.merge(products, on="product_id")[["product_name", "predicted_reorder_prob"]]


if __name__ == "__main__":
    orders, prior, train_labels, products = load_data()
    features, user_product = create_profile(orders, prior)
    user_clusters = pd.read_csv("DataSet/user_clusters.csv")
    X, y = build_model(orders, train_labels, user_product, features)
    model, X_set, f1 = train_model(X, y)

    user_id = 1
    recs = recommender(
        user_id,
        user_product,
        features[["user_id", "user_total_orders_clean"]],
        features[["product_id", "product_total_purchases_clean", "product_avg_cart_order_clean"]],
        model,
        products)

    print("product recommendations:\n", recs)
