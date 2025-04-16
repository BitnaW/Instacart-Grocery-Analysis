import numpy as np
import matplotlib.pyplot as plt
from load_csv import *
from init_model import *
from create_profile import *
from recommender import *

if __name__ == "__main__":
    orders, prior, train_labels, products = load_data()
    features, user_product = create_profile(orders, prior)
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

    
    print("Top recommendations:\n", recs)
