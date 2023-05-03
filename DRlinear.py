import numpy as np
from statsmodels.api import OLS
from statsmodels.tools import add_constant

class DRLinear:
    """Validation test for CATE models. Estimates OLS of doubly-robust (DR) outcomes on CATE estimates (plus a
    constant), and tests whether coefficient on CATE estimates are different than 0. Can oly be used for binary
    treatment.

    Parameters
    ----------
    cate_model: estimator
        The CATE estimator used to fit the treatment effect to features. Must be already fitted in training sample and
        able to implement the `predict' method
    model_y_zero: estimator
        Nuisance model estimator used to fit the outcome to features in the untreated group. Must be already fitted in
        training sample and able to implement the `predict' method
    model_y_one: estimator
        Nuisance model estimator used to fit the outcome to features in the treated group. Must be already fitted in
        training sample and able to implement the `predict' method
    model_t: estimator
        Nuisance model estimator used to fit the treatment assignment to features.  Must be already fitted in
        training sample and able to implement the `predict' method


    References
    ----------
    V. Chernochukov et al.
    Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments
    arXiv preprint arXiv:1712.04802, 2022.
    `<https://arxiv.org/abs/1712.04802>`_

    """
    def __init__(
        self,
        cate_model,
        model_y_zero,
        model_y_one,
        model_t,
    ):
        self.cate_model = cate_model
        self.model_y_zero = model_y_zero
        self.model_y_one = model_y_one
        self.model_t = model_t

    def calculate_dr_outcomes(
        self,
        X,
        D,
        y
    ):

        """
        Calculates doubly robust outcomes using nuisance models.

        Parameters
        ----------
        X: (n x k) matrix or vecotr of length n
            Features used to predict outomes/treatments in model_y_zero, model_y_one, and model_t
        D: vector of length n
            Treatment assignment
        y: vector of length n
            Outcomes

        Returns
        -------
        dr: vector of length n
            Doubly-robust outcomes
        """

        reg_zero_preds= self.model_y_zero.predict(X) # Predict y(0)
        reg_one_preds = self.model_y_one.predict(X) # Predict y(1)
        reg_preds = reg_zero_preds * (1 - D) + reg_one_preds * D # Prediction of y(D)
        prop_preds = self.model_t.predict(X) # Predict D

        # Calculate doubly-robust outcome
        dr = reg_one_preds - reg_zero_preds
        # Reiz representation, clip denominator at 0.01
        reisz = (D - prop_preds) / np.clip(prop_preds * (1 - prop_preds), .01, np.inf)
        dr += (y - reg_preds) * reisz

        return dr

    def fit(
        self,
        X,
        D,
        Y,
        Z
    ):
        """

        Fits OLS of doubly-robust outcomes on a constant and CATE prediction.

        Parameters
        ----------
        X: (n x k) matrix or vector of length n
            Features used in nuisance models (model_y_zero, model_y_one, model_t)
        D: vector of length n
            Treatment assignments
        y: vecotr of length n
            Outcomes
        Z: (n x k) matrix or vector of length n
            Features used in the CATE model

        Returns
        -------
        self
        """

        # Calculate doubly-robust outcomes
        self.dr_outcomes_ = self.calculate_dr_outcomes(X, D, Y)

        # Generate CATE predictions
        self.cate_predictions_ = self.cate_model.predict(Z)

        # Fit OLS of DR outcomes on constant and CATE predictions
        self.res = OLS(self.dr_outcomes_, add_constant(self.cate_predictions_)).fit()

        self.params = self.res.params

        return self
