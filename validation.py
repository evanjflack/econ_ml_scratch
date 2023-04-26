import numpy as np
from sklearn.base import clone
from statsmodels.api import OLS
from statsmodels.tools import add_constant



class DRLinear:
    """Validation test for CATE models. Estimates OLS of doubly-robust outcomes on CATE estimates, and tests whether
    coefficient on CATE estimates are different than 0. Can oly be used for binary treatment.



    Parameters
    ----------
    cate_model: estimator
        The estimator for fitting the treatment effect to the features (e.g. DRlearner). Must have a predict method.

    model_y_zero: estimator
        The estimator for fitting the response variable to features for untreated units

    model_y_one: estimator
        The estimator for fitting the response variable to features for treated units

    model_t: estimator
        The estimator for fitting the treatment status to the features


    References
    ----------


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

        Parameters
        ----------

        X: (n x k array)
            Covariates used to predict response/treatment
        D: n x 1 vector
            Treatment indicators
        Y: vector, outcome


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

