from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer

from sklearn.metrics import fbeta_score, make_scorer

from shared import *
#from ml_metrics import rmsle

def predict_linear():
    training = pd.read_csv(TRAIN_FEATURES_CSV)
    test = pd.read_csv(TEST_FEATURES_CSV)
    
    loss = make_scorer(root_mean_squared_logarithmic_error_loss_func, greater_is_better=False)
     
    clf = LinearRegression()
    training_values = training[TOTAL_TRAINING_FEATURE_COLUMNS].values
    # imp = Imputer(axis=0, missing_values='NaN', strategy='mean', verbose=0)
    # imp.fit(training_values)
    
    # cap the prediction outliers (seems to help)
    CAP_PREDICTION_VALUE = 10
    training.loc[training[TARGET_COLUMN[0]] > CAP_PREDICTION_VALUE, TARGET_COLUMN[0]] = CAP_PREDICTION_VALUE
    
    clf.fit(training_values, training[TARGET_COLUMN].values)
    
    test_values = test[TOTAL_TRAINING_FEATURE_COLUMNS].values
    # imp.fit(test_values)
    predictions = [i[0] for i in clf.predict(test_values)]
    
    submission = pd.DataFrame({"id":test['id'], "Demanda_uni_equil": predictions})
    submission.loc[submission['Demanda_uni_equil'] < 0,'Demanda_uni_equil'] = 0
    submission.to_csv(PREDICTION_CSV, index=False, cols=['id', 'Demanda_uni_equil'])

def predict_xgboost():
    
    import xgboost as xgb
    
    training = pd.read_csv(TRAIN_FEATURES_CSV)
    test = pd.read_csv(TEST_FEATURES_CSV)
    
    # cap the prediction outliers (seems to help with linear, not with xgboost)
    #CAP_PREDICTION_VALUE = 10
    #training.loc[training[TARGET_COLUMN[0]] > CAP_PREDICTION_VALUE, TARGET_COLUMN[0]] = CAP_PREDICTION_VALUE
    
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
    
    #print ('RMSLE Score:', rmsle(y_test, preds))
    
    submission = pd.DataFrame({"id":test['id'], "Demanda_uni_equil": predictions})
    submission.loc[submission['Demanda_uni_equil'] < 0,'Demanda_uni_equil'] = 0
    submission.to_csv(PREDICTION_CSV, index=False, cols=['id', 'Demanda_uni_equil'])

if __name__ == "__main__":
    # predict_linear()
    predict_xgboost()
    
    sub_df = pd.read_csv(PREDICTION_CSV)
    print(sub_df.shape)
    print(sub_df.isnull().sum())
    print(sub_df.describe())