import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import sys
import random

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import wordnet
from blaze.expr.reductions import nrows

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold, train_test_split

from shared import *

N_ROWS = 5000000
N_FOLDS = 5
RANDOM_STATE = 451

if __name__ == "__main__":
    
    # get the products and city/state dataframe
    products_df = pd.read_pickle(PRODUCTS_PKL)
    cs_df = pd.read_pickle(CITY_STATE_PKL)
    
    # get a random subsample of the demand csv file
    n = sum(1 for line in open(TRAINING_CSV)) - 1 #number of records in file (excludes header)
    skip = sorted(random.sample(xrange(1,n + 1), n - N_ROWS)) #the 0-indexed header will not be included in the skip list
    df = pd.read_csv(TRAINING_CSV, 
                dtype  = {'Semana': 'int8',
                    'Agencia_ID':'int32',
                    'Canal_ID': 'int32',
                    'Ruta_SAK':'int32',
                    'Cliente_ID':'int32',
                    'Producto_ID':'int32',
                    'Demanda_uni_equil':'int32'},
                skiprows=skip)
    
    # split into K-Folds
    kf = KFold(n=df.shape[0], n_folds=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_indices, test_indices in kf:
        # split the df into train/test
        df_train = df.iloc[train_indices]
        df_test = df.iloc[test_indices]
        
        # create the meta data using the train set (test must not be used)
        median_demand = df_train['Demanda_uni_equil'].median()
        demand_cust_median_df  = get_median_cust_demand_df(df_train)
        demand_cust_prod_median_df  = get_median_cust_demand_df(df_train)
        
        # load the dataframes with the new meta data
        df_train = combine_dataframes(df_train, products_df, cs_df, demand_cust_median_df, demand_cust_prod_median_df, median_demand)
        df_test = combine_dataframes(df_test, products_df, cs_df, demand_cust_median_df, demand_cust_prod_median_df, median_demand)
                
        # cap the prediction to 10
        CAP_PREDICTION_VALUE = 10
        df_train.loc[df_train[TARGET_COLUMN[0]] > CAP_PREDICTION_VALUE, TARGET_COLUMN[0]] = CAP_PREDICTION_VALUE
        
        clf = LinearRegression()
        clf.fit(df_train[TOTAL_TRAINING_FEATURE_COLUMNS], df_train[TARGET_COLUMN])