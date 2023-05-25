import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from statsmodels.api import OLS
from statsmodels.tools import add_constant


class DRtester:

    """Validation tests for CATE models. Includes the best linear predictor (BLP) test as in Chernozhukov et al. (2022)
    as well as the calibration test in Dwivedi et al. (2020). Can handle multiple categorical treatments.

    Parameters
    ----------
    reg_outcome: estimator
        Nuisance model estimator used to fit the outcome to features. Must be able to implement `fit' and predict
        methods

    reg_t: estimator
        Nuisance model estimator used to fit the treatment assignment to features. Must be able to implement `fit'
        method and either `predict' (in the case of binary treatment) or `predict_proba' method (in the case of multiple
        categorical treatments).

    n_splits: integer
        Number of splits used to generate cross-validated predictions (default = 5)


    References
    ----------
    R. Dwivedi et al.
    Stable Discovery of Interpretable Subgroups via Calibration in Causal Studies
    International Statistical Review (2020), 88, S1, S135–S178 doi:10.1111/insr.12427

    V. Chernozhukov et al.
    Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments
    arXiv preprint arXiv:1712.04802, 2022.
    `<https://arxiv.org/abs/1712.04802>`_

    """

    def __init__(
        self,
        reg_outcome,
        reg_t,
        n_splits=5
    ):
        self.reg_outcome = reg_outcome
        self.reg_t = reg_t
        self.n_splits = n_splits
        self.dr_train = None
        self.cate_preds_train = None
        self.cate_preds_val = None
        self.dr_val = None

    # Fits nusisance and CATE
    def fit_nuisance(
        self,
        Xval,
        Dval,
        yval,
        Xtrain = None,
        Dtrain = None,
        ytrain = None,
    ):

        """

        Generates nuisance predictions either by (1) cross-fitting in the validation sample, or (2) fitting in the
        training sample and applying to the validation sample. If Xtrain, Dtrain, and ytrain are all not None,
        then option (2) will be implemented, otherwise, option (1) will be implemented. In order to use the
        `evaluate_cal' method then Xtrain, Dtrain, and ytrain must all be specified.

        Parameters
        ----------
        Xval: (n_val x k) matrix or vector of length n
            Features used in nuisance models for validation sample
        Dval: vector of length n_val
            Treatment assignment of validation sample. Control status must be minimum value. It is recommended to have
            the control status be equal to 0, and all other treatments integers starting at 1.
        Yval: vector of length n_val
            Outcomes for the validation sample
        Xtrain: (n_train x k) matrix or vector of length n
            Features used in nuisance models for training sample
        Dtrain: vector of length n_train
            Treatment assignment of training sample. Control status must be minimum value. It is recommended to have
            the control status be equal to 0, and all other treatments integers starting at 1.
        Ytrain: vector of length n_train
            Outcomes for the training sample

        Returns
        ------
        self

        """

        self.Dval = Dval

        # Unique treatments (ordered, includes control)
        self.tmts = np.sort(np.unique(Dval))

        # Number of treatments (excluding control)
        self.n_treat = len(self.tmts) - 1

        # Indicator for whether
        self.fit_on_train = (Xtrain is not None) and (Dtrain is not None) and (ytrain is not None)

        if self.fit_on_train:
            # Get DR outcomes in training sample
            reg_preds_train, prop_preds_train = self.fit_nuisance_cv(Xtrain, Dtrain, ytrain)
            self.dr_train = self.calculate_dr_outcomes(self.n_treat, Dtrain, ytrain, reg_preds_train, prop_preds_train)

            # Get DR outcomes in validation sample
            reg_preds_val, prop_preds_val = self.fit_nuisance_train(Xtrain, Dtrain, ytrain, Xval)
            self.dr_val = self.calculate_dr_outcomes(self.n_treat, Dval, yval, reg_preds_val, prop_preds_val)

            self.Dtrain = Dtrain

        else:
            reg_preds_val, prop_preds_val = self.fit_nuisance_cv(Xval, Dval, yval)
            self.dr_val = self.calculate_dr_outcomes(self.n_treat, Dval, yval, reg_preds_val, prop_preds_val)

        self.ate_val = np.mean(self.dr_val, axis=0)

        return self

    def fit_cate(self, reg_cate, Zval, Ztrain = None):

        if (Ztrain is None) & self.fit_on_train:
            raise Exception("Nuisance models fit on training sample but Ztrain not specified")

        if (Ztrain is not None) & (self.fit_on_train == False):
            raise Exception("Nuisance models fit fit (cv) in validation sample but Ztrain is specified")

        if Ztrain is not None:
            self.cate_preds_train = self.fit_cate_cv(self.n_splits, reg_cate, Ztrain, self.Dtrain, self.dr_train)
            self.cate_preds_val = self.fit_cate_train(reg_cate, Ztrain, Zval)
        else:
            self.cate_preds_val = self.fit_cate_cv(self.n_splits, reg_cate, Zval, self.Dval, self.dr_val)

        return self

    def evaluate_cal(self, n_groups=4):

        if (self.cate_preds_val is None) or (self.cate_preds_train is None) or (self.dr_val is None):
            raise Exception("Must fit CATE before evaluating")

        if self.dr_train is None:
            raise Exception("Must fit nuisance/CATE models on training sample data to use calibration test")

        self.cal_r_squared = np.zeros(self.n_treat)
        self.df_plot = pd.DataFrame()
        for k in range(self.n_treat):

            cuts = np.quantile(self.cate_preds_train[:, k], np.linspace(0, 1, n_groups + 1))
            probs = np.zeros(n_groups)
            g_cate = np.zeros(n_groups)
            se_g_cate = np.zeros(n_groups)
            gate = np.zeros(n_groups)
            se_gate = np.zeros(n_groups)
            for i in range(n_groups):
                # Assign units in validation set to groups
                ind = (self.cate_preds_val[:, k] >= cuts[i]) & (self.cate_preds_val[:, k] <= cuts[i + 1])
                # Proportion of validations set in group
                probs[i] = np.mean(ind)
                # Group average treatment effect (GATE) -- average of DR outcomes in group
                gate[i] = np.mean(self.dr_val[ind, k])
                se_gate[i] = np.std(self.dr_val[ind, k]) / np.sqrt(np.sum(ind))
                # Average of CATE predictions in group
                g_cate[i] = np.mean(self.cate_preds_val[ind, k])
                se_g_cate[i] = np.std(self.cate_preds_val[ind, k]) / np.sqrt(np.sum(ind))

            # Calculate group calibration score
            cal_score_g = np.sum(abs(gate - g_cate) * probs)
            # Calculate overall calibration score
            cal_score_o = np.sum(abs(gate - self.ate_val[k]) * probs)
            # Calculate R-square calibration score
            self.cal_r_squared[k] = 1 - (cal_score_g / cal_score_o)

            df_plot1 = pd.DataFrame({'ind': np.array(range(n_groups)),
                                     'gate': gate, 'se_gate': se_gate,
                                    'g_cate': g_cate, 'se_g_cate': se_g_cate})
            df_plot1['tmt'] = self.tmts[k + 1]
            self.df_plot = pd.concat((self.df_plot, df_plot1))

        return self

    def plot_cal(self, tmt):
        df = self.df_plot
        df = df[df.tmt == tmt].copy()
        rsq = round(self.cal_r_squared[np.where(self.tmts == tmt)[0][0] - 1], 3)
        df['95_err'] = 1.96 * df['se_gate']
        fig = df.plot(kind='scatter',
            x='g_cate',
            y='gate',
            yerr='95_err',
            xlabel = 'Group Mean CATE',
            ylabel = 'GATE',
            title=f"Treatment = {tmt}, Calibration R^2 = {rsq}")

        return fig

    def evaluate_blp(self):
        if (self.cate_preds_val is None) or (self.dr_val is None):
            raise Exception("Must fit CATE before evaluating")

        if self.n_treat == 1:  # binary treatment
            reg = OLS(self.dr_val, add_constant(self.cate_preds_val)).fit()
            params = [reg.params[1]]
            errs = [reg.bse[1]]
            pvals = [reg.pvalues[1]]
        else:  # categorical treatment
            params = []
            errs = []
            pvals = []
            for k in range(self.n_treat):  # run a separate regression for each
                reg = OLS(self.dr_val[:, k], add_constant(self.cate_preds_val[:, k])).fit(cov_type = 'HC1')
                params.append(reg.params[1])
                errs.append(reg.bse[1])
                pvals.append(reg.pvalues[1])

        self.blp_res = pd.DataFrame(
            {'treatment': self.tmts[1:], 'blp_est': params, 'blp_se': errs, 'blp_pval': pvals}
        ).round(3)

        return self

    def evaluate_all(self, n_groups=4):

        self.evaluate_blp()
        self.evaluate_cal(n_groups)
        self.evaluate_qini()

        self.df_res = self.blp_res.merge(self.qini_res, on='treatment')
        self.df_res['cal_r_squared'] = np.around(self.cal_r_squared, 3)

        return self

    # Fits nuisance in train, predicts in validation
    def fit_nuisance_train(self, Xtrain, Dtrain, ytrain, Xval):

        # Fit propensity in treatment
        reg_t_fitted = self.reg_t.fit(Xtrain, Dtrain)
        # Predict propensity scores
        if self.n_treat == 1:
            prop_preds = reg_t_fitted.predict(Xval)
        else:
            prop_preds = reg_t_fitted.predict_proba(Xval)

        # Possible treatments (need to allow more than 2)
        tmts = np.sort(np.unique(Dtrain))
        n = Xval.shape[0]
        k = len(tmts)
        reg_preds = np.zeros((n, k))
        for i in range(k):
            reg_outcome_fitted = self.reg_outcome.fit(Xtrain[Dtrain == tmts[i]], ytrain[Dtrain == tmts[i]])
            reg_preds[:, i] = reg_outcome_fitted.predict(Xval)

        return reg_preds, prop_preds

    # CV nuisance predictions
    def fit_nuisance_cv(self, X, D, y, shuffle=True, random_state=123):

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)
        splits = list(cv.split(X, D))

        if self.n_treat == 1:
            prop_preds = cross_val_predict(self.reg_t, X, D, cv=splits)
        else:
            prop_preds = cross_val_predict(self.reg_t, X, D, cv=splits, method='predict_proba')

        # Predict outcomes
        # T-learner logic
        tmts = np.sort(np.unique(D))
        N = X.shape[0]
        K = len(tmts)
        reg_preds = np.zeros((N, K))
        for k in range(K):
            for train, test in splits:
                reg_outcome_fitted = self.reg_outcome.fit(X[train][D[train] == tmts[k]], y[train][D[train] == tmts[k]])
                reg_preds[test, k] = reg_outcome_fitted.predict(X[test])

        return reg_preds, prop_preds

    # Calculates DR outcomes
    @staticmethod
    def calculate_dr_outcomes(
        n_treat,
        D: np.array,
        y: np.array,
        reg_preds,
        prop_preds
    ) -> np.array:

        if n_treat == 1:  # if treatment is binary
            reg_preds_chosen = np.sum(reg_preds * np.column_stack((D, 1 - D)), axis=1)

            # Calculate doubly-robust outcome
            dr = reg_preds[:, 1] - reg_preds[:, 0]
            # Reiz representation, clip denominator at 0.01
            reisz = (D - prop_preds) / np.clip(prop_preds * (1 - prop_preds), .01, np.inf)
            dr += (y - reg_preds_chosen) * reisz
        else:  # if treatment is categorical
            # treat each treatment as a separate regression
            # here, prop_preds should be a matrix
            # with rows corresponding to units and columns corresponding to treatment statuses
            dr_vec = []
            d0_mask = np.where(D == 0, 1, 0)
            y_dr_0 = reg_preds[:, 0] + (d0_mask / np.clip(prop_preds[:, 0], .01, np.inf)) * (y - reg_preds[:, 0])
            for k in np.sort(np.unique(D)):  # pick a treatment status
                if k > 0:  # make sure it is not control
                    dk_mask = np.where(D == k, 1, 0)
                    y_dr_k = reg_preds[:, k] + (dk_mask / np.clip(prop_preds[:, k], .01, np.inf)) * (y - reg_preds[:, k])
                    dr_k = y_dr_k - y_dr_0  # this is an n x 1 vector
                    dr_vec.append(dr_k)
            dr = np.column_stack(dr_vec)  # this is an n x k matrix

        return dr

    @staticmethod
    def cate_fit_predict(n_treat, reg_cate, train, test, dr):

        if np.ndim(test) == 1:
            test = test.reshape(-1, 1)

        if np.ndim(train) == 1:
            train = train.reshape(-1, 1)

        if n_treat == 1:
            reg_cate_fitted = reg_cate.fit(train, dr)
            cate_preds = reg_cate_fitted.predict(test)
        else:
            cate_preds = []
            for k in range(n_treat):  # fit a separate cate model for each treatment status?
                reg_cate_fitted = reg_cate.fit(train, dr[:, k])
                cate_preds.append(reg_cate_fitted.predict(test))

            cate_preds = np.column_stack(cate_preds)

        return cate_preds

    # Fits CATE in training, predicts in validation
    def fit_cate_train(self, reg_cate, Ztrain, Zval):
        return self.cate_fit_predict(self.n_treat, reg_cate, Ztrain, Zval, self.dr_train)

    # CV prediction of CATEs
    @staticmethod
    def fit_cate_cv(n_splits, reg_cate, Z, D, dr, shuffle=True, random_state=712):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        splits = list(cv.split(Z, D))

        if np.ndim(Z) == 1:
            Z = Z.reshape(-1, 1)

        N = Z.shape[0]
        tmts = np.sort(np.unique(D))
        K = len(tmts)
        cate_preds = np.zeros((N, K - 1))

        for k in range(K - 1):
            cate_preds[:, k] = cross_val_predict(reg_cate, Z, dr[:, k], cv = splits)

        return cate_preds

    @staticmethod
    def calc_qini_coeff(cate_preds_train, cate_preds_val, dr_val, percentiles):
        qs = np.percentile(cate_preds_train, percentiles)
        toc, toc_std, group_prob = np.zeros(len(qs)), np.zeros(len(qs)), np.zeros(len(qs))
        toc_psi = np.zeros((len(qs), dr_val.shape[0]))
        n = len(dr_val)
        ate = np.mean(dr_val)
        for it in range(len(qs)):
            inds = (qs[it] <= cate_preds_val)  # group with larger CATE prediction than the q-th quantile
            group_prob = np.sum(inds) / n  # fraction of population in this group
            toc[it] = group_prob * (
                    np.mean(dr_val[inds]) - ate)  # tau(q) = q * E[Y(1) - Y(0) | tau(X) >= q[it]] - E[Y(1) - Y(0)]
            toc_psi[it, :] = (dr_val - ate) * (inds - group_prob) - toc[it]  # influence function for the tau(q)
            toc_std[it] = np.sqrt(np.mean(toc_psi[it] ** 2) / n)  # standard error of tau(q)

        qini_psi = np.sum(toc_psi[:-1] * np.diff(percentiles).reshape(-1, 1) / 100, 0)
        qini = np.sum(toc[:-1] * np.diff(percentiles) / 100)
        qini_stderr = np.sqrt(np.mean(qini_psi ** 2) / n)

        return qini, qini_stderr

    def evaluate_qini(self, percentiles=np.linspace(5, 95, 50)):

        if (self.cate_preds_val is None) or (self.cate_preds_train is None) or (self.dr_val is None):
            raise Exception("Must fit CATE before evaluating")

        if self.n_treat == 1:
            qini, qini_err = self.calc_qini_coeff(
                self.cate_preds_train,
                self.cate_preds_val,
                self.dr_val,
                percentiles
            )
            qinis = [qini]
            errs = [qini_err]
        else:
            qinis = []
            errs = []
            for k in range(self.n_treat):
                qini, qini_err = self.calc_qini_coeff(
                    self.cate_preds_train[:, k],
                    self.cate_preds_val[:, k],
                    self.dr_val[:, k],
                    percentiles
                )

                qinis.append(qini)
                errs.append(qini_err)

        pvals = [st.norm.sf(abs(q / e)) for q, e in zip(qinis, errs)]

        self.qini_res = pd.DataFrame(
            {'treatment': self.tmts[1:], 'qini_coeff': qinis, 'qini_se': errs, 'qini_pval': pvals},
        ).round(3)

        return self
