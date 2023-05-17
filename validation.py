import numpy as np
from statsmodels.api import OLS
from statsmodels.tools import add_constant
from sklearn.model_selection import cross_val_predict, StratifiedKFold


# Fits nuisance in train, predicts in validation
def fit_nuisance_train(reg_outcome, reg_t, Xtrain, Dtrain, ytrain, Xval, Dval):

    treatments = np.unique(D)
    n = Xval.shape[0]
    k = len(treatments)
    reg_preds = np.zeros((n, k))
    for i in treatments:
        reg_outcome_fitted = reg_outcome().fit(Xtrain[Dtrain == i], ytrain[Dtrain == i])
        reg_preds[:, i] = reg_outcome_fitted.predict(Xval)

    reg_t_fitted = reg_t().fit(Xtrain, Dtrain)
    prop_preds = reg_t_fitted.predict(Xval)

    return reg_preds, prop_preds

# CV nuisance predictions
def fit_nuisance_cv(reg_outcome, reg_t, X, D, y, n_splits = 5, shuffle = True, random_state = 712):

    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(cv.split(X, D))

    tmts = np.unique(D)
    n = X.shape[0]
    k = len(tmts)
    reg_preds = np.zeros((n, k))

    for i in range(len(tmts)):
        for train, test in splits:
            reg_outcome_fitted = reg_outcome().fit(X.iloc[train][D[train] == tmts[i]], y[train][D[train] == tmts[i]])
            reg_preds[test, i] = reg_outcome_fitted.predict(X.iloc[test])

    prop_preds = cross_val_predict(reg_t(), X, D, cv=splits)

    return reg_preds, prop_preds

# Fits CATE in training, predicts in validation
def fit_cate_train(reg_cate, dr_train, Ztrain, Zval):

    reg_cate_fitted = reg_cate.fit(Ztrain, dr_train)
    cate_preds = reg_cate_fitted.predict(Zval)

    return cate_preds

# CV prediction of CATEs
def fit_cate_cv(reg_cate, dr, Z, D, shuffle = True, random_state = 712):

    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(cv.split(Z, D))

    n = X.shape[0]
    cate_preds = np.zeros(n)

    for train, test in splits:
        reg_cate_fitted = reg_cate.fit(Z.iloc[train], dr[train])
        cate_preds[test] = reg_cate_fitted.predict(Z.iloc[test])

    return cate_preds


def calculate_dr_outcomes(
        X: np.array,
        D: np.array,
        y: np.array,
        model_y_zero,
        model_y_one,
        model_t
):
    """
    Calculates doubly robust outcomes using fitted nuisance models.

    Parameters
    ----------
    X: (n x k) matrix or vecotr of length n
        Features used to predict outomes/treatments in model_y_zero, model_y_one, and model_t
    D: vector of length n
        Treatment assignment
    y: vector of length n
        Outcomes
    model_y_zero: estimator
        Nuisance model estimator used to fit the outcome to features in the untreated group. Must be already fitted in
        training sample and able to implement the `predict' method
    model_y_one: estimator
        Nuisance model estimator used to fit the outcome to features in the treated group. Must be already fitted in
        training sample and able to implement the `predict' method
    model_t: estimator
        Nuisance model estimator used to fit the treatment assignment to features.  Must be already fitted in
        training sample and able to implement the `predict' method

    Returns
    -------
    dr: vector of length n
        Doubly-robust outcomes
    """

    reg_zero_preds = model_y_zero.predict(X)  # Predict y(0)
    reg_one_preds = model_y_one.predict(X)  # Predict y(1)
    reg_preds = reg_zero_preds * (1 - D) + reg_one_preds * D  # Prediction of y(D)
    prop_preds = model_t.predict(X)  # Predict D

    # Calculate doubly-robust outcome
    dr = reg_one_preds - reg_zero_preds
    # Reiz representation, clip denominator at 0.01
    reisz = (D - prop_preds) / np.clip(prop_preds * (1 - prop_preds), .01, np.inf)
    dr += (y - reg_preds) * reisz

    return dr


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
    V. Chernozhukov et al.
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


    def fit(
        self,
        Xval,
        Dval,
        yval,
        Xtrain = None,
        Dtrain = None,
        ytrain = None

            if (Xtrain != None) & (Dtrain != None) & (ytrain != None):



        ):

    def fit(
        self,
        X: np.array,
        D: np.array,
        y: np.array,
        Z: np.array
    ):
        """

        Fits OLS of doubly-robust outcomes on a constant and CATE prediction.

        Parameters
        ----------
        X: (n x k) matrix or vector of length n
            Features used in nuisance models (model_y_zero, model_y_one, model_t)
        D: vector of length n
            Treatment assignments
        y: vector of length n
            Outcomes
        Z: (n x k) matrix or vector of length n
            Features used in the CATE model

        Returns
        -------
        self
        """

        # Calculate doubly-robust outcomes
        self.dr_outcomes_ = calculate_dr_outcomes(
            model_y_zero=self.model_y_zero,
            model_y_one=self.model_y_one,
            model_t=self.model_t,
            X=X,
            D=D,
            y=y
        )

        # Generate CATE predictions
        self.cate_predictions_ = self.cate_model.predict(Z)

        # Fit OLS of DR outcomes on constant and CATE predictions
        self.res = OLS(self.dr_outcomes_, add_constant(self.cate_predictions_)).fit()

        self.params = self.res.params

        self.bse = self.res.bse

        return self


class cal_scorer:
    """
    Validation score based on calibration of predicted group average treatment effects (GATE) from a CATE model to the
    actual GATE (average of doubly robust outcomes). Groups are quantiles of the CATE prediction in a different sample.

    First, quantile cut points of the CATE prediction are defined in the test set for equal-sized groups. Units in the
    validation set are then assigned to one of each of these groups (k). Within each group, the average of the CATE
    prediction E[s(Z) | k] and the average of the doubly-robust outcomes E[Ydr, | k] is calculated along with the
    overall ATE E[Ydr].

    The within group cal-score is then defined as:

    .. math::
        cal-score_g := \sum_{k} \pi(k) \cdot \Big|E[s(Z) | k] - E[Ydr | k]\Big|

    where :math:`\pi(k)` is the proportion of units in the validation set in group k.

    The overall cal-score is defined as:

     .. math::
        cal-score_o := \sum_{k} \pi(k) \cdot |E[s(Z) | k] - E[Ydr]|

    Then calibration R squared score is:

    .. math::
       R^2_c := 1 - (cal-score-g/cal-score-o)

    The calibration R score can take any (real) value less than or equal to 1, with values closer to 1 indicating a
    better calibrated model. It can be interpreted as the degree to which the CATE estimator explains the variability
    of the CATE with respect to the partition, in comparison to the best constant model.

    Parameters
    ----------
    cate_model: estimator
        The CATE estimator used to fit the treatment effect to features. Must be already fitted in a training sample and
        able to implement the `predict' method.
    model_y_zero: estimator
        Nuisance model estimator used to fit the outcome to features in the untreated group. Must be already fitted in
        a training sample and able to implement the `predict' method.
    model_y_one: estimator
        Nuisance model estimator used to fit the outcome to features in the treated group. Must be already fitted in
        a training sample and able to implement the `predict' method.
    model_t: estimator
        Nuisance model estimator used to fit the treatment assignment to features.  Must be already fitted in
        a training sample and able to implement the `predict' method.
    n_groups: integer
        Number of groups to bin the validation sample into (using quantiles defined in the test sample). For example,
        if n_groups = 4, cuts will be based on the 25th, 50th, and 75th percentiles.


    References
    ----------
    R. Dwivedi et al.
    Stable Discovery of Interpretable Subgroups via Calibration in Causal Studies
    International Statistical Review (2020), 88, S1, S135–S178 doi:10.1111/insr.12427
    """

    def __init__(
            self,
            cate_model,
            model_y_zero,
            model_y_one,
            model_t,
            n_groups: int
    ):
        self.cate_model = cate_model
        self.model_y_zero = model_y_zero
        self.model_y_one = model_y_one
        self.model_t = model_t
        self.n_groups = n_groups

    def score(
            self,
            Xval: np.array,
            Dval: np.array,
            Yval: np.array,
            Zval: np.array,
            Ztest: np.array
    ):
        """
        Parameters
        ----------
        Xval: (n_val x k) matrix or vector of length n
            Features used in nuisance models for validation sample
        Dval: vector of length n_val
            Treatment assignment of validation sample
        Yval: vector of length n_val
            Outcomes for the validation sample
        Zval: (n_val x k) matrix or vector of length n
            Features used in the CATE model for the validation sample
        Ztest: (n_val x k) matrix or vector of length n_val
            Features used in the CATE model for the test sample (in which quantiles for groups will be defined)


        Returns
        -------
        self
        """
        # Calculate DR outcomes in validation set
        self.dr_outcomes_val_ = calculate_dr_outcomes(
            model_y_zero=self.model_y_zero,
            model_y_one=self.model_y_one,
            model_t=self.model_t,
            X=Xval,
            D=Dval,
            y=Yval
        )

        # Predict CATE in validation set
        self.cate_preds_val_ = self.cate_model.predict(Zval)

        # PRedict CATE in test set
        cate_preds_test = self.cate_model.predict(Ztest)

        # Define CATE quantile cut points in test set
        cuts = np.quantile(cate_preds_test, np.linspace(0, 1, self.n_groups + 1))

        # Calculate average DR outcome and average CATE prediction for each group (in validation set)
        self.probs = np.zeros(self.n_groups)
        self.g_cate = np.zeros(self.n_groups)
        self.se_g_cate = np.zeros(self.n_groups)
        self.gate = np.zeros(self.n_groups)
        self.se_gate = np.zeros(self.n_groups)

        for i in range(self.n_groups):
            # Assign units in validation set to groups
            ind = (self.cate_preds_val_ >= cuts[i]) & (self.cate_preds_val_ <= cuts[i + 1])
            # Proportion of validations set in group
            self.probs[i] = np.mean(ind)
            # Group average treatment effect (GATE) -- average of DR outcomes in group
            self.gate[i] = np.mean(self.dr_outcomes_val_[ind])
            self.se_gate[[i]] = np.std(self.dr_outcomes_val_[ind]) / np.sqrt(np.sum(ind))
            # Average of CATE predictions in group
            self.g_cate[i] = np.mean(self.cate_preds_val_[ind])
            self.se_g_cate[[i]] = np.std(self.cate_preds_val_[ind]) / np.sqrt(np.sum(ind))

        # Calculate overall ATE
        self.ate = np.mean(self.dr_outcomes_val_)

        # Calculate group calibration score
        cal_score_g = np.sum(abs(self.gate - self.g_cate) * self.probs)
        # Calculate overall calibration score
        cal_score_o = np.sum(abs(self.gate - self.ate) * self.probs)
        # Calculate R-square calibration score
        self.r_squared_cal = 1 - (cal_score_g / cal_score_o)

        return self
