import pandas as pd

def load_data():
    orders = pd.read_csv("DataSet/orders.csv")
    prior = pd.read_csv("DataSet/order_products__prior.csv")
    train_labels = pd.read_csv("DataSet/order_products__train.csv")
    products = pd.read_csv("DataSet/products.csv")
    return orders, prior, train_labels, products
