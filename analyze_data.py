import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import sys

from shared import *

# https://www.kaggle.com/anokas/grupo-bimbo-inventory-demand/exploratory-data-analysis/comments
# https://www.kaggle.com/vykhand/grupo-bimbo-inventory-demand/exploring-products/discussion
# https://www.kaggle.com/lyytinen/grupo-bimbo-inventory-demand/basic-preprocessing-for-products/output

# https://www.kaggle.com/kobakhit/grupo-bimbo-inventory-demand/xgboost


def nl():
    print('\n')


def list_files():
    for f in os.listdir(DATA_DIR):
        print(f.ljust(30) + str(round(os.path.getsize(DATA_DIR + f) / 1000000, 2)) + 'MB')


def train_test_analyze():
    #df_train = pd.read_csv(TRAINING_CSV, nrows=500000)
    #df_test = pd.read_csv(TEST_CSV, nrows=500000)
    df_train = pd.read_csv(TRAINING_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    nl()
    print('Size of training set: ' + str(df_train.shape))
    print(' Size of testing set: ' + str(df_test.shape))
    
    nl()
    print('Columns in train: ' + str(df_train.columns.tolist()))
    print(' Columns in test: ' + str(df_test.columns.tolist()))
    
    nl()
    print(df_train.describe())
    print(df_test.describe())

    nl()
    print(df_train.head())


def get_product_agg(df_train, cols):
    agg  = df_train.groupby(['Semana', 'Producto_ID'], as_index=False).agg(['count','sum', 'min', 'max','median','mean'])
    agg.columns  =  ['_'.join(col).strip() for col in agg.columns.values]
    del(df_train)
    return agg


def train_analysis():
    # I want to investigate how demand changes based on the weeks
    # df_train = pd.read_csv(TRAINING_CSV, nrows=20000000)
    # agg = get_product_agg(df_train, ['Demanda_uni_equil',])
    
    df_train = pd.read_csv(TRAINING_CSV, nrows=200000)
    demand_median_df  = df_train.groupby(['Cliente_ID','Producto_ID'], as_index=False)['Demanda_uni_equil'].median()
    demand_median_df.columns = ['Cliente_ID','Producto_ID','Demanda_uni_equil_median',]
    print(demand_median_df.head())
    
    df_train = pd.merge(df_train, demand_median_df, on=['Cliente_ID', 'Producto_ID'])
    print(df_train.head())
    

def view_train_rows(n=20):
    df_train = pd.read_csv(TRAINING_CSV, nrows=n)
    print(df_train)


def view_target():
    df_train = pd.read_csv(TRAINING_CSV)
    target = df_train['Demanda_uni_equil'].tolist()
    print(Counter(target).most_common(10))
    
def create_features(load_from_pkl=True):
    
    # get the products dateframe
    if load_from_pkl:
        products = pd.read_pickle(PRODUCTS_PKL)
    else:
        products = analyze_products()
        products.to_pickle(PRODUCTS_PKL)
    
    # get the city/state dataframe
    if load_from_pkl:
        cs_df = pd.read_pickle(CITY_STATE_PKL)
    else:
        cs_df = analyze_city_state()
        cs_df.to_pickle(CITY_STATE_PKL)

    """
    # read in s
    df_train = pd.read_csv(TRAINING_CSV, iterator=True, chunksize=200000)
    
    first = True
    for chunk in df_train:
        
        chunk_merge = pd.merge(products, chunk, on='Producto_ID')
        print(chunk_merge.shape)
        
        features = chunk_merge.get(TRAINING_FEATURE_COLUMNS + vectorizer.get_feature_names() + TARGET_COLUMN)
        print(features.shape)
        
        if first:
            features.to_csv(FEATURE_CSV)
            first = False
        else:
            features.to_csv(FEATURE_CSV, mode='a', header=False)
    """
    
    # Gets the training data, I'm setting the dtype to use less memory
    df_train = pd.read_csv(TRAINING_CSV, 
        dtype  = {'Semana': 'int8',
                  'Agencia_ID':'int32',
                  'Canal_ID': 'int32',
                  'Ruta_SAK':'int32',
                  'Cliente_ID':'int32',
                  'Producto_ID':'int32',
                  'Demanda_uni_equil':'int32'})
    
    # calculate the median demand
    median_demand = df_train['Demanda_uni_equil'].median()
    
    # calculate the median demand from a client
    if load_from_pkl:
        demand_cust_median_df = pd.read_pickle(FULL_TRAIN_DEMAND_CUST_MEDIAN_PKL)
    else:
        demand_cust_median_df  = get_median_cust_demand_df(df_train)
        demand_cust_median_df.to_pickle(FULL_TRAIN_DEMAND_CUST_MEDIAN_PKL)
    
    # calculate the median demand from a client/product
    if load_from_pkl:
        demand_cust_prod_median_df = pd.read_pickle(FULL_TRAIN_DEMAND_CUST_PROD_MEDIAN_PKL)
    else:
        demand_cust_prod_median_df  = get_median_cust_prod_demand_df(df_train)
        demand_cust_prod_median_df.to_pickle(FULL_TRAIN_DEMAND_CUST_PROD_MEDIAN_PKL)
    
    # calculate the median agency demand
    demand_agen_median_df = get_median_agen_demand_df(df_train)
    
    # calculate the median agency/product/demand
    demand_agen_prod_median_df = get_median_agen_prod_demand_df(df_train)
    
    # calculate the median customer/product/agency/demand
    demand_cust_prod_agen_median_df = get_median_cust_prod_agen_demand_df(df_train)
    
    # calculate the previous weeks agency demand
    demand_semana_agen_median_df = get_previous_week_agen_demand_df(df_train)
    
    # calculate the previous weeks agency/product demand
    demand_semana_agen_prod_median_df = get_previous_week_agen_prod_demand_df(df_train)
    
    """
    # the median returns doesn't help
    median_returns = df_train['Dev_uni_proxima'].median()
    returns_cust_median_df  = df_train.groupby(['Cliente_ID',], as_index=False)['Dev_uni_proxima'].median()
    returns_cust_median_df.columns = ['Cliente_ID','Dev_uni_proxima_median',] 
    
    # this doesn't help either
    # add sales from previous week information, per customer, per product
    demand_cust_week_prod_median_df  = df_train.groupby(['Cliente_ID','Semana','Producto_ID'], as_index=False)['Demanda_uni_equil'].median()
    demand_cust_week_prod_median_df.columns = ['Cliente_ID','Semana','Producto_ID','Demanda_uni_equil_median_cust_week_prod',]
    demand_cust_week_prod_median_df['Semana'] = demand_cust_week_prod_median_df['Semana'] + 1
    """
    
    # load the dataframes with the new meta data
    df_train = combine_dataframes(df_train, products, cs_df, demand_cust_median_df, demand_cust_prod_median_df, demand_cust_prod_agen_median_df, demand_agen_median_df, demand_agen_prod_median_df, demand_semana_agen_median_df, demand_semana_agen_prod_median_df, median_demand)
    df_train.get(TOTAL_TRAINING_FEATURE_COLUMNS + TARGET_COLUMN).to_csv(TRAIN_FEATURES_CSV, index=False)
    
    # Gets the test data, I'm setting the dtype to use less memory
    df_test = pd.read_csv(TEST_CSV, 
        dtype  = {'id': 'int32',
                  'Semana': 'int8',
                  'Agencia_ID':'int32',
                  'Canal_ID': 'int32',
                  'Ruta_SAK':'int32',
                  'Cliente_ID':'int32',
                  'Producto_ID':'int32'})
    
    # create the test set dataframe
    df_test = combine_dataframes(df_test, products, cs_df, demand_cust_median_df, demand_cust_prod_median_df, demand_cust_prod_agen_median_df, demand_agen_median_df, demand_agen_prod_median_df, demand_semana_agen_median_df, demand_semana_agen_prod_median_df, median_demand)
    df_test.get(TOTAL_TEST_FEATURE_COLUMNS).to_csv(TEST_FEATURES_CSV, index=False)
    
    print(df_test.shape)


def test_pd():
    client_one  =  pd.read_csv(CLIENT_CSV)
    client_two  =  pd.read_csv(CLIENT_CSV)
    
    print(client_one.shape)
    
    result = pd.merge(client_one, client_two)
    print(result.shape)


def compare_product_lists():
    df_train = pd.read_csv(TRAINING_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    product_dict = {}
    for v in df_train['Producto_ID'].values:
        product_dict[v] = 1
        
    missing_products = []
    
    for v in df_test['Producto_ID'].values:
        if not product_dict.has_key(v):
            missing_products.append(v)
            
    print(len(set(df_test['Producto_ID'].values)))
    print(set(missing_products))
    print(len(set(missing_products)))
    

if __name__ == "__main__":
    #list_files()
    #train_test_analyze()
    #view_train_rows()
    #view_target()
    #train_analysis()
    
    #analyze_products()
    #analyze_city_state()
    #test_pd()
    #compare_product_lists()
    
    create_features(load_from_pkl=False)