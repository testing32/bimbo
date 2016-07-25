import numpy as np

class XGBoostRegressionSKLearnWrapper():
    
    def __init__(self, params, boost_rounds=2000):
        self.params = params
        self.boost_rounds = boost_rounds
        
    def fit(self, x, y):
        import xgboost as xgb
        y = [float(value) for value in y]
        dtrain = xgb.DMatrix(x, label=y)
        watchlist  = [(dtrain,'train')]
        
        self.clf = xgb.train(self.params, dtrain, self.boost_rounds, watchlist, verbose_eval=False)

    def predict(self, x):
        import xgboost as xgb
        return self.clf.predict(xgb.DMatrix(x))



class XGBoostClassifierSKLearnWrapper():
    
    def __init__(self, params, boost_rounds=2000):
        self.params = params
        self.boost_rounds = boost_rounds
        
    def fit(self, x, y):
        import xgboost as xgb
        y = [self.prob_transform(float(value)) for value in y]
        dtrain = xgb.DMatrix(x, label=y)
        watchlist  = [(dtrain,'train')]
        
        self.clf = xgb.train(self.params, dtrain, self.boost_rounds, watchlist, verbose_eval=False)

    def predict(self, x):
        import xgboost as xgb
        return [self.prob_transform(value) for value in self.clf.predict(xgb.DMatrix(x))]

    def predict_proba(self, x):
        import xgboost as xgb
        return np.array([[value, 1-value] for value in self.clf.predict(xgb.DMatrix(x))])

    def prob_transform(self, value):
        if value >= 0.5:
            return 1
        else:
            return 0