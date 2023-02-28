# Installation des librairies et des fonctions usuelles de python                                 

import csv
import pandas as pd

customers = pd.read_csv('D:\Openclassrooms\Projets\projet 05\Data\olist_customers_dataset.csv', sep=',')
geolocalisation = pd.read_csv('D:\Openclassrooms\Projets\projet 05\Data\olist_geolocation_dataset.csv', sep=',')
order_items = pd.read_csv('D:\Openclassrooms\Projets\projet 05\Data\olist_order_items_dataset.csv', sep=',')
order_payments = pd.read_csv('D:\Openclassrooms\Projets\projet 05\Data\olist_order_payments_dataset.csv', sep=',')
order_reviews = pd.read_csv('D:\Openclassrooms\Projets\projet 05\Data\olist_order_reviews_dataset.csv', sep=',')
orders = pd.read_csv('D:\Openclassrooms\Projets\projet 05\Data\olist_orders_dataset.csv', sep=',')
products = pd.read_csv('D:\Openclassrooms\Projets\projet 05\Data\olist_products_dataset.csv', sep=',')
sellers = pd.read_csv('D:\Openclassrooms\Projets\projet 05\Data\olist_sellers_dataset.csv', sep=',')
translation = pd.read_csv('D:\Openclassrooms\Projets\projet 05\Data\product_category_name_translation.csv', sep=',')

list_data = {'customers': customers, 
              'geolocalisation': geolocalisation,
              'order_items': order_items,
              'order_payments': order_payments,
              'order_reviews': order_reviews,
              'orders': orders,
              'products': products,
              'sellers': sellers,
              'translation': translation}

def dataframes_info(list_data):
    list_info = []
    for name, df in list_data.items():
        list_info.append([name, df.shape[0], df.shape[1],
                          df.count().sum(),
                          df.isna().mean().mean()*100,
                          df.duplicated().sum()])

    df_info = pd.DataFrame(list_info, columns=['name', 'rows', 'columns', 'count',
                                               'NaNs (%)', 'duplicates'])
    return df_info


print(dataframes_info(list_data))