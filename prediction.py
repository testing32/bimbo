import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer

from sklearn.metrics import fbeta_score, make_scorer

from shared import *

def root_mean_squared_logarithmic_error_loss_func(ground_truths, predictions):
    ground_truths = np.asarray(ground_truths)
    predictions = np.asarray(predictions)
    
    n = len(ground_truths)
    diff = (np.log(predictions+1) - np.log(ground_truths+1))**2
    print(diff, n, np.sum(diff))
    return np.sqrt(np.sum(diff)/n)

def predict():
    training = pd.read_csv(TRAIN_FEATURES_CSV)
    
    loss = make_scorer(root_mean_squared_logarithmic_error_loss_func, greater_is_better=False)
     
    clf = LinearRegression()
    training_values = training[TOTAL_TRAINING_FEATURE_COLUMNS].values
    # imp = Imputer(axis=0, missing_values='NaN', strategy='mean', verbose=0)
    # imp.fit(training_values)
    
    # cap the prediction outliers (seems to help)
    CAP_PREDICTION_VALUE = 10
    training.loc[training[TARGET_COLUMN[0]] > CAP_PREDICTION_VALUE, TARGET_COLUMN[0]] = CAP_PREDICTION_VALUE
    
    clf.fit(training_values, training[TARGET_COLUMN].values)
    
    test = pd.read_csv(TEST_FEATURES_CSV)
    test_values = test[TOTAL_TRAINING_FEATURE_COLUMNS].values
    # imp.fit(test_values)
    predictions = [i[0] for i in clf.predict(test_values)]
    
    submission = pd.DataFrame({"id":test['id'], "Demanda_uni_equil": predictions})
    submission.loc[submission['Demanda_uni_equil'] < 0,'Demanda_uni_equil'] = 0
    submission.to_csv(PREDICTION_CSV, index=False, cols=['id', 'Demanda_uni_equil'])

if __name__ == "__main__":
    predict()
    
    sub_df = pd.read_csv(PREDICTION_CSV)
    print(sub_df.shape)
    print(sub_df.isnull().sum())
    print(sub_df.describe())