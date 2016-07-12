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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold, train_test_split

from shared import *

# the validation error is higher than the full dataset error by about 5%
N_ROWS = 2000000
N_FOLDS = 5
RANDOM_STATE = 451

def predict_linear(df_train, df_test):
    # train the classifier
    clf = LinearRegression()
    clf.fit(df_train[TOTAL_TRAINING_FEATURE_COLUMNS], df_train[TARGET_COLUMN])
    
    # predict test
    predictions = [i[0] for i in clf.predict(df_test[TOTAL_TRAINING_FEATURE_COLUMNS])]
    
    # collect the rmsl
    #rmsl = root_mean_squared_logarithmic_error_loss_func(df_test[TARGET_COLUMN].values, predictions)
    #loss.append(rmsl)
    result = evalerror2(predictions, df_test[TARGET_COLUMN].values)
    print(result)
    return result[1]

def predict_xgboost(df_train, df_test):
    
    import xgboost as xgb
    
    training = pd.read_csv(TRAIN_FEATURES_CSV)
    test = pd.read_csv(TEST_FEATURES_CSV)
    
    # cap the prediction outliers (seems to help)
    CAP_PREDICTION_VALUE = 10
    training.loc[training[TARGET_COLUMN[0]] > CAP_PREDICTION_VALUE, TARGET_COLUMN[0]] = CAP_PREDICTION_VALUE
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1729)
    
    params = {"objective": "reg:linear",
          "booster" : "gbtree",
          # 'eval_metric': 'ndcg',
          'lambda': 1,
          'alpha': 0,
          "eta": 0.01,
          "gamma": .5,
          "max_depth": 4,
          "subsample": 0.7,
          "colsample_bytree": 0.4,
          "min_child_weight": 7,
          "silent": 1,
          "thread": -1,
          "nthread": 20,
          "seed": 1301
          } # 0.873502225    0.946107784    0.580526638    0.719544592

    
    num_boost_round = 150
    cv_num_round = 2

    dtrain = xgb.DMatrix(training[TOTAL_TRAINING_FEATURE_COLUMNS], label=training[TARGET_COLUMN])
    dtest = xgb.DMatrix(test[TOTAL_TRAINING_FEATURE_COLUMNS])
    
    test_preds = np.zeros(test.shape[0])    
    watchlist  = [(dtrain,'train')]
    
    xg_classifier = xgb.train(params, dtrain, num_boost_round, watchlist, feval=evalerror, early_stopping_rounds=20, verbose_eval=10)
    predictions = xg_classifier.predict(dtest, ntree_limit=xg_classifier.best_iteration)
    
    result = evalerror(predictions, df_test[TARGET_COLUMN].values)
    print(result)
    return result[1]


def predict_train_test(df_train, df_test, model_prediction=predict_linear):

    load_training_test_df(df_train, df_test, products_df, cs_df)
                
    # cap the prediction to 10
    CAP_PREDICTION_VALUE = 10
    df_train.loc[df_train[TARGET_COLUMN[0]] > CAP_PREDICTION_VALUE, TARGET_COLUMN[0]] = CAP_PREDICTION_VALUE
    
    model_prediction(df_train, df_test)
    
    
if __name__ == "__main__":
    
    # error with best generated stuff
    # ('error', 0.7939924617954093)
    
    # product clustering added
    # ('error', 0.7877704717125115)
    
    # added extra mean
    # ('error', 0.787393188036041)
    
    # added agencia id
    # ('error', 0.7876189211318351)
    
    # switched from client based means to agencia based means
    # ('error', 0.7793947204950505)
    
    # added all of the means
    # ('error', 0.8020385276209636)
    
    # analyze_products().to_pickle(PRODUCTS_PKL)
    # analyze_city_state().to_pickle(CITY_STATE_PKL)
    
    # get the products and city/state dataframe
    products_df = pd.read_pickle(PRODUCTS_PKL)
    cs_df = pd.read_pickle(CITY_STATE_PKL)
    
    # get a random subsample of the demand csv file
    n = 74180464 # sum(1 for line in open(TRAINING_CSV)) - 1 #number of records in file (excludes header)
    skip = sorted(random.sample(xrange(1,n + 1), n - N_ROWS)) #the 0-indexed header will not be included in the skip list
    df = pd.read_csv(TRAINING_CSV, 
                dtype  = {'Semana': 'int8',
                    'Agencia_ID':'int32',
                    'Canal_ID': 'int32',
                    'Ruta_SAK':'int32',
                    'Cliente_ID':'int32',
                    'Producto_ID':'int32',
                    'Demanda_uni_equil':'int32'}, skiprows=skip)
    
    # split into K-Folds
    kf = KFold(n=df.shape[0], n_folds=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_indices, test_indices in kf:
        # split the df into train/test and predict
        predict_train_test(df.iloc[train_indices], df.iloc[test_indices])
        
        # I'm only doing this once because the folds are so close to the same values
        break        
        