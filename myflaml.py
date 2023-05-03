import numpy as np
from flaml import AutoML
from sklearn.base import BaseEstimator, clone
import warnings
warnings.simplefilter('ignore')
###################################
# AutoML models
###################################

# FLAML models don't return "self" at end of fit. We create this wrapper.


class AutoMLWrap(BaseEstimator):

    def __init__(self, *, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model_ = clone(self.model)
        self.model_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def auto_reg(X, y, *, groups=None, n_splits=5, split_type='auto', time_budget=60, verbose=0):
    X = np.array(X)
    automl = AutoML(task='regression', time_budget=time_budget, early_stop=True,
                    eval_method='cv', n_splits=n_splits, split_type=split_type,
                    metric='mse', verbose=verbose, estimator_list = ['xgboost'])
    inds = np.arange(X.shape[0])
    np.random.shuffle(inds)
    if groups is None:
        automl.fit(X[inds], y[inds])
    else:
        automl.fit(X[inds], y[inds], groups=groups[inds])
    best_est = automl.best_estimator
    return lambda: AutoMLWrap(model=clone(automl.best_model_for_estimator(best_est)))


class AutoMLWrapCLF(BaseEstimator):

    def __init__(self, *, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model_ = clone(self.model)
        self.model_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model_.predict_proba(X)[:, 1]

# By default if we use metric='mse' for a classification task, then flaml
# will use `predict` instead of `predict_proba`. We define a custom mse
# loss for classification.
def clf_mse(
        X_val, y_val, estimator, labels,
        X_train, y_train, weight_val=None, weight_train=None,
        *args,):
    val_loss = np.mean((estimator.predict_proba(X_val)[:, 1] - y_val)**2)
    return val_loss, {"val_loss": val_loss}

def auto_clf(X, y, *, groups=None, n_splits=5, split_type='auto', time_budget=60, verbose=0):
    X = np.array(X)
    automl = AutoML(task='classification', time_budget=time_budget, early_stop=True,
                    eval_method='cv', n_splits=n_splits, split_type=split_type,
                    metric=clf_mse, verbose=verbose, estimator_list = ['xgboost'])
    inds = np.arange(X.shape[0])
    np.random.shuffle(inds)
    if groups is None:
        automl.fit(X[inds], y[inds])
    else:
        automl.fit(X[inds], y[inds], groups=groups[inds])
    best_est = automl.best_estimator
    return lambda: AutoMLWrapCLF(model=clone(automl.best_model_for_estimator(best_est)))