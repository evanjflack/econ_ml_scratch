import numpy as np
from sklearn.base import clone
from statsmodels.api import OLS
from statsmodels.tools import add_constant



class DRLinear:
    """
    x = DRLinear(cate, zero, one, t)
    x_fitted = x.fit(X,Y,D,Z)

    x.model ...
    """
    def __init__(
        self,
        cate_model,
        model_y_zero,
        model_y_one,
        model_t,
    ):
        self.cate_model = clone(cate_model, safe=False)
        self.model_y_zero = clone(model_y_zero, safe=False)
        self.model_y_one = clone(model_y_one, safe=False)
        self.model_t = clone(model_t, safe=False)

    def calculate_dr_outcomes(
        self,
        X,
        D,
        y
    ):
        """

        :param X: covariate data
        :param D: treatment assignment
        :param y: outcomes
        :return:
        """
        reg_zero_preds_t = self.model_y_zero.predict(X)
        reg_one_preds_t = self.model_y_one.predict(X)
        reg_preds_t = reg_zero_preds_t * (1 - D) + reg_one_preds_t * D
        prop_preds = self.model_t.predict(X)

        dr = reg_one_preds_t - reg_zero_preds_t
        reisz = (D - prop_preds) / np.clip(prop_preds * (1 - prop_preds), .09, np.inf)
        dr += (y - reg_preds_t) * reisz

        return dr

    def fit(
        self,
        X,
        D,
        Y,
        Z
    ):
        """

        :param X: covariate data
        :param D: treatment assignment
        :param y: outcomes
        :param Z: subsetted covariates on which to test heterogeneity
        :return:
        """
        self.dr_outcomes_ = self.calculate_dr_outcomes(X, D, Y)

        self.cate_predictions_ = self.cate_model.predict(Z)

        self.model = OLS(self.dr_outcomes_, add_constant(self.cate_predictions_)).fit()

        return self

