from sklearn.base import BaseEstimator


class Identity(BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    def predict(self, X):
        return X #np.full(shape=X.shape[0], fill_value=self.param)