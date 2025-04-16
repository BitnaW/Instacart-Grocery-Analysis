from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

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
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.21).astype(int)

    f1 = f1_score(y_val, y_pred)
    print(f"F1-score: {f1:.4f}")

    return model, X, f1
