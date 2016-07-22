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

def test_scalar():
    from sklearn.preprocessing.data import MinMaxScaler, StandardScaler
    scalar = StandardScaler()
    
    training = pd.read_csv(TRAIN_FEATURES_CSV, nrows=200000)
    test = pd.read_csv(TEST_FEATURES_CSV)
    
    # normalize the values
    for column in TOTAL_TRAINING_FEATURE_COLUMNS:
        training[column] = scalar.fit_transform(training[column])
        test[column] = scalar.transform(test[column])

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
        products_df = pd.read_pickle(PRODUCTS_PKL)
    else:
        products_df = analyze_products()
        products_df.to_pickle(PRODUCTS_PKL)
    
    # get the city/state dataframe
    if load_from_pkl:
        cs_df = pd.read_pickle(CITY_STATE_PKL)
    else:
        cs_df = analyze_city_state()
        cs_df.to_pickle(CITY_STATE_PKL)
    
    # Gets the training data, I'm setting the dtype to use less memory
    df_train = pd.read_csv(TRAINING_CSV, 
        dtype  = {'Semana': 'int8',
                  'Agencia_ID':'int32',
                  'Canal_ID': 'int32',
                  'Ruta_SAK':'int32',
                  'Cliente_ID':'int32',
                  'Producto_ID':'int32',
                  'Demanda_uni_equil':'int32'})
    
    # Gets the test data, I'm setting the dtype to use less memory
    df_test = pd.read_csv(TEST_CSV, 
        dtype  = {'id': 'int32',
                  'Semana': 'int8',
                  'Agencia_ID':'int32',
                  'Canal_ID': 'int32',
                  'Ruta_SAK':'int32',
                  'Cliente_ID':'int32',
                  'Producto_ID':'int32'})
    
    df_train, df_test = load_training_test_df(df_train, df_test, products_df, cs_df)
    
    df_train.get(TOTAL_TRAINING_FEATURE_COLUMNS + TARGET_COLUMN).to_csv(TRAIN_FEATURES_CSV, index=False)    
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
    
def group_up_agencies():
    df_train = pd.read_csv(TRAINING_CSV)#, nrows=2000000)
    
    new_df = df_train.groupby('Agencia_ID').Canal_ID.nunique().to_frame()
    new_df.columns = ['unique_canal_per_agency',]
    new_df['Agencia_ID'] = new_df.index
    print(new_df.describe())
    
    new_df = df_train.groupby('Cliente_ID').Canal_ID.nunique().to_frame()
    new_df.columns = ['unique_canal_per_client',]
    new_df['Cliente_ID'] = new_df.index
    print(new_df.describe())
    
    #df = pd.merge(df_train, new_df, on=['Agencia_ID',])
    
    """ very low std
    new_df = df_train.groupby('Cliente_ID').Agencia_ID.nunique().to_frame()
    new_df.columns = ['unique_agencia_per_client',]
    new_df['Cliente_ID'] = new_df.index
    print(new_df.describe())
    
    new_df = df_train.groupby('Agencia_ID').Cliente_ID.nunique().to_frame()
    new_df.columns = ['unique_clients_per_agency',]
    new_df['Agencia_ID'] = new_df.index
    print(new_df.describe())
    """
    
    new_df = df_train.groupby('Cliente_ID').Producto_ID.nunique().to_frame()
    new_df.columns = ['unique_products_per_client',]
    new_df['Cliente_ID'] = new_df.index
    print(new_df.describe())

    new_df = df_train.groupby('Agencia_ID').Producto_ID.nunique().to_frame()
    new_df.columns = ['unique_products_per_agency',]
    new_df['Agencia_ID'] = new_df.index
    print(new_df.describe())
            
    new_df = df_train.groupby('Producto_ID').Cliente_ID.nunique().to_frame()
    new_df.columns = ['unique_clients_per_product',]
    new_df['Producto_ID'] = new_df.index
    print(new_df.describe())

    new_df = df_train.groupby('Producto_ID').Agencia_ID.nunique().to_frame()
    new_df.columns = ['unique_agencies_per_product',]
    new_df['Producto_ID'] = new_df.index
    print(new_df.describe())

def log_mean():
    df_train = pd.read_csv(TRAINING_CSV, nrows=9000000)
    
    log_result = np.log(df_train['Demanda_uni_equil'] + 1)
    mean_result = np.mean(log_result)
    log_mean = exp_result = np.exp(mean_result) - 1

    df_train['log_mean'] = log_mean

    mean = df_train['Demanda_uni_equil'].mean()    
    df_train['mean'] = mean
    
    median = df_train['Demanda_uni_equil'].median()
    df_train['median'] = median
    
    print(evalerror2(df_train['log_mean'], df_train['Demanda_uni_equil']))
    print(evalerror2(df_train['mean'], df_train['Demanda_uni_equil']))
    print(evalerror2(df_train['median'], df_train['Demanda_uni_equil']))

def get_log_mean_right():
    df = pd.read_csv(TRAINING_CSV, nrows=200000)
    
    demand_agen_median_df  = df.groupby(['Agencia_ID',], as_index=False)['Demanda_uni_equil'].median()
    demand_agen_median_df.columns = ['Agencia_ID','Demanda_uni_equil_median_agen',]
    
    demand_agen_log_mean_df = df.groupby(['Agencia_ID',], as_index=False).Demanda_uni_equil.agg(calc_log_mean)
    demand_agen_log_mean_df.columns = ['Agencia_ID','Demanda_uni_equil_log_mean_agen',]
    
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
    
    #create_features()
    #group_up_agencies()
    
    #get_log_mean_right()
    test_scalar()