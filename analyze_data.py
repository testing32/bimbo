import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import wordnet
from blaze.expr.reductions import nrows

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


def analyze_products():
    products  =  pd.read_csv(PRODUCT_CSV)
    
    # gets the first part of the product name and stops at the first digit (weight)
    products['short_name'] = products.NombreProducto.str.extract('^(\D*)', expand=False)
    products['short_name_encoded'] = LabelEncoder().fit_transform(products['short_name'])
    
    # starts from the end and gets the first letters at the 2nd to last
    products['brand'] = products.NombreProducto.str.extract('^.+\s(\D+) \d+$', expand=False)
    products['brand_encoded'] = LabelEncoder().fit_transform(products['brand'])
    
    # get the weights
    w = products.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)
    products['weight'] = w[0].astype('float')*w[1].map({'Kg':1000, 'g':1})
    
    # get the number of pieces
    products['pieces'] =  products.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')
            
    stemmer = SnowballStemmer("spanish")
    products['short_name_processed'] = (products['short_name']
                                        .map(lambda x: " ".join([stemmer.stem(i) for i in x.lower()
                                                                 .split() if i not in stopwords.words("spanish")])))
    
    
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 1000,
                             dtype=np.uint8) 
    product_bag_words = vectorizer.fit_transform(products.short_name_processed).toarray()
    print(product_bag_words.shape)
    print(products.shape)
    
    avg_brand_pieces_df = products.loc[np.isnan(products['pieces']) == False, ['pieces','brand_encoded']].groupby(['brand_encoded',], as_index=False)['pieces'].mean()
    avg_brand_weight_df = products.loc[np.isnan(products['weight']) == False, ['weight','brand_encoded']].groupby(['brand_encoded',], as_index=False)['weight'].mean()
    avg_brand_weight_df.columns = ['brand_encoded','avg_brand_weight']
    
    products = pd.merge(products, avg_brand_weight_df, on='brand_encoded', how='left')
    
    # fill in the nan values with the mean of the column
    products['pieces'] = products['pieces'].fillna(1)
    
    # this was not an improvement
    # products.loc[np.isnan(products['weight']), 'weight'] = products.loc[np.isnan(products['weight']), 'avg_brand_weight'] 
    
    products = products.fillna(products.mean())
    
    products_bagofwords = pd.concat([products.get(PRODUCT_FEATURE_COLUMNS), 
                               pd.DataFrame(product_bag_words, 
                                            columns= vectorizer.get_feature_names(), index = products.index)], axis=1).to_sparse()
    
    #print(products_bagofwords.shape)
    #print(products.describe())
    
    return products

    
def analyze_city_state():
    cs_df  =  pd.read_csv(CITY_STATE_CSV)
    cs_df['city_state_encoded'] = LabelEncoder().fit_transform(cs_df['Town'] + cs_df['State'])
    cs_df['state_encoded'] = LabelEncoder().fit_transform(cs_df['State'])
    
    return cs_df

    
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
    
    # calculate the median demand from a client
    median_demand = df_train['Demanda_uni_equil'].median()    
    if load_from_pkl:
        demand_cust_median_df = pd.read_pickle(FULL_TRAIN_DEMAND_CUST_MEDIAN_PKL)
    else:
        demand_cust_median_df  = df_train.groupby(['Cliente_ID',], as_index=False)['Demanda_uni_equil'].median()
        demand_cust_median_df.columns = ['Cliente_ID','Demanda_uni_equil_median',]
        demand_cust_median_df.to_pickle(FULL_TRAIN_DEMAND_CUST_MEDIAN_PKL)
    
    # calculate the median demand from a client/product
    if load_from_pkl:
        demand_cust_prod_median_df = pd.read_pickle(FULL_TRAIN_DEMAND_CUST_PROD_MEDIAN_PKL)
    else:
        demand_cust_prod_median_df  = df_train.groupby(['Cliente_ID','Producto_ID'], as_index=False)['Demanda_uni_equil'].median()
        demand_cust_prod_median_df.columns = ['Cliente_ID','Producto_ID','Demanda_uni_equil_median_prod',]
        demand_cust_prod_median_df.to_pickle(FULL_TRAIN_DEMAND_CUST_PROD_MEDIAN_PKL)
    
    # create the training set dataframe
    #df_train = pd.merge(products_bagofwords, df_train, on='Producto_ID')
    df_train = pd.merge(products.get(PRODUCT_FEATURE_COLUMNS + ['short_name_encoded',]), df_train, on='Producto_ID')
    df_train = pd.merge(cs_df.get(CITY_STATE_FEATURE_COLUMNS + ['Agencia_ID',]), df_train, on='Agencia_ID')
    df_train = pd.merge(df_train, demand_cust_median_df, on='Cliente_ID')
    df_train = pd.merge(df_train, demand_cust_prod_median_df, on=['Cliente_ID', 'Producto_ID'])
    
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
    df_test = pd.merge(products.get(PRODUCT_FEATURE_COLUMNS + ['short_name_encoded',]), df_test, on='Producto_ID')
    df_test = pd.merge(cs_df.get(CITY_STATE_FEATURE_COLUMNS + ['Agencia_ID',]), df_test, on='Agencia_ID')
    
    df_test = pd.merge(df_test, demand_cust_median_df, on='Cliente_ID', how='left')
    df_test['Demanda_uni_equil_median'] = df_test['Demanda_uni_equil_median'].fillna(median_demand)
    
    df_test = pd.merge(df_test, demand_cust_prod_median_df, on=['Cliente_ID', 'Producto_ID'], how='left')
    df_test.loc[np.isnan(df_test['Demanda_uni_equil_median_prod']), 'Demanda_uni_equil_median_prod'] = df_test.loc[np.isnan(df_test['Demanda_uni_equil_median_prod']), 'Demanda_uni_equil_median'] 
    
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
    train_test_analyze()
    #view_train_rows()
    #view_target()
    #train_analysis()
    
    #analyze_products()
    #analyze_city_state()
    #test_pd()
    #compare_product_lists()
    
    #create_features()