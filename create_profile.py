def create_profile(orders, prior):
    prior = prior.merge(orders[["order_id", "user_id", "order_number"]], on="order_id")

    user_product = prior.groupby(["user_id", "product_id"]).agg(
        user_prod_order_count=("order_id", "count"),
        user_prod_avg_cart_order=("add_to_cart_order", "mean"),
    ).reset_index()

    user_features = prior.groupby("user_id").agg(
        user_total_orders=("order_number", "max"),
    ).reset_index()

    product_features = prior.groupby("product_id").agg(
        product_total_purchases=("order_id", "count"),
        product_avg_cart_order=("add_to_cart_order", "mean")
    ).reset_index()

    user_product = user_product.rename(columns={
        "user_prod_order_count": "user_prod_order_count_clean",
        "user_prod_avg_cart_order": "user_prod_avg_cart_order_clean"
    })
    user_features_clean = user_features.rename(columns={
        "user_total_orders": "user_total_orders_clean"
    })
    product_features_clean = product_features.rename(columns={
        "product_total_purchases": "product_total_purchases_clean",
        "product_avg_cart_order": "product_avg_cart_order_clean"
    })

    features = user_product.merge(user_features_clean, on="user_id", how="left")
    features = features.merge(product_features_clean, on="product_id", how="left")

    return features, user_product
