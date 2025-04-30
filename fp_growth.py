import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    order_products = pd.read_csv(r"DataSet/order_products__prior.csv")
    products = pd.read_csv(r"DataSet/products.csv")

    order_products = order_products.merge(products, on='product_id', how='left')

    order_sample = order_products[order_products['order_id'] <= 16384]

    basket = (order_sample
              .groupby(['order_id', 'product_name'])['product_name']
              .count().unstack().fillna(0).astype(int))
    basket = basket > 0

    frequent_itemsets = fpgrowth(basket, min_support=0.01, use_colnames=True)

    frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)

    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    rules.sort_values(by='confidence', ascending=False, inplace=True)
    print(rules)

    top_items = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]
    top_items = top_items.head(10).copy()
    top_items['itemset_str'] = top_items['itemsets'].apply(lambda x: ', '.join(list(x)))

    plt.figure(figsize=(10, 6))
    plt.barh(top_items['itemset_str'], top_items['support'])
    plt.xlabel('Support')
    plt.title('Top 10 Frequent Itemsets (size > 1)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
