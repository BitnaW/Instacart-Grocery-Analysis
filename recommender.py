def recommender(user_id, user_product, user_features, product_features, model, products):
    user_data = user_product[user_product.user_id == user_id]
 
    user_data = user_data.merge(user_features, on="user_id", how="left")
    user_data = user_data.merge(product_features, on="product_id", how="left")

    user_data = user_data.groupby(["user_id", "product_id"], as_index=False).agg({
        'user_prod_order_count_clean': 'max',
        'user_prod_avg_cart_order_clean': 'mean',
        'user_total_orders_clean': 'max',
        'product_total_purchases_clean': 'sum',
        'product_avg_cart_order_clean': 'mean'
    })

    #print("used COLUMNS:", user_data.columns)
    #print("user data AFTER: ")
    #print(user_data)

    rec_user = user_data.drop(columns=["user_id", "product_id"])

    print("trained on: ", list(model.feature_names_in_))
    rec_user = rec_user.reindex(columns=model.feature_names_in_, fill_value=0)

    calc_probs = model.predict_proba(rec_user)[:, 1]
    user_data["predicted_reorder_prob"] = calc_probs

    top_products = user_data.sort_values(by="predicted_reorder_prob", ascending=False).head(11)
    return top_products.merge(products, on="product_id")[["product_name", "predicted_reorder_prob"]]


