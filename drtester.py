import numpy as np
import pandas as pd
from statsmodels.api import OLS
from statsmodels.tools import add_constant
from sklearn.model_selection import cross_val_predict, StratifiedKFold


class DRtester:

    def __init__(
        self,
        reg_outcome,
        reg_t,
        n_splits=5
    ):
        self.reg_outcome = reg_outcome
        self.reg_t = reg_t
        self.n_splits = n_splits

    # Fits nusisance and CATE
    def fit(
        self,
        reg_cate,
        Xval,
        Dval,
        yval,
        Zval,
        Xtrain = None,
        Dtrain = None,
        ytrain = None,
        Ztrain = None,
    ):
        self.n_treat = Dval.max()

        if (Xtrain is not None) and (Dtrain is not None) and (ytrain is not None) and (Ztrain is not None):
            reg_preds_train, prop_preds_train = self.fit_nuisance_cv(Xtrain, Dtrain, ytrain)
            self.dr_train = self.calculate_dr_outcomes(Dtrain, ytrain, reg_preds_train, prop_preds_train)

            reg_preds_val, prop_preds_val = self.fit_nuisance_train(Xtrain, Dtrain, ytrain, Xval)
            self.dr_val = self.calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val)

            self.cate_preds_val = self.fit_cate_train(reg_cate, Ztrain, Zval)
            # self.cate_preds_train = self.fit_cate_cv(reg_cate, self.dr_train, Ztrain, Dtrain)

        else:
            reg_preds_val, prop_preds_val = self.fit_nuisance_cv(Xval, Dval, yval)
            self.dr_val = self.calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val)
            self.cate_preds_val = self.fit_cate_cv(reg_cate, Zval, Dval)

        return self

    def evaluate_blp(self):
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
                reg = OLS(self.dr_val[:, k], add_constant(self.cate_preds_val[:, k])).fit()
                params.append(reg.params[1])
                errs.append(reg.bse[1])
                pvals.append(reg.pvalues[1])

        self.blp_res = pd.DataFrame({'Estimate': params, 'Std. Err': errs, 'p-Value': pvals})

        return self

    # Fits nuisance in train, predicts in validation
    def fit_nuisance_train(self, Xtrain, Dtrain, ytrain, Xval):

        # Possible treatments (need to allow more than 2)
        tmts = np.sort(Dtrain.unique())
        n = Xval.shape[0]
        k = len(tmts)
        reg_preds = np.zeros((n, k))
        for i in range(k):
            reg_outcome_fitted = self.reg_outcome().fit(Xtrain[Dtrain == tmts[i]], ytrain[Dtrain == tmts[i]])
            reg_preds[:, i] = reg_outcome_fitted.predict(Xval)

        reg_t_fitted = self.reg_t().fit(Xtrain, Dtrain)
        prop_preds = reg_t_fitted.predict(Xval)

        return reg_preds, prop_preds

    # CV nuisance predictions
    def fit_nuisance_cv(self, X, D, y, shuffle=True, random_state=712):

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)
        splits = list(cv.split(X, D))

        tmts = np.sort(np.unique(D))
        n = X.shape[0]
        k = len(tmts)
        reg_preds = np.zeros((n, k))

        for i in range(k):
            for train, test in splits:
                reg_outcome_fitted = self.reg_outcome().fit(X.iloc[train][D[train] == tmts[i]], y[train][D[train] == tmts[i]])
                reg_preds[test, i] = reg_outcome_fitted.predict(X.iloc[test])

        prop_preds = cross_val_predict(self.reg_t(), X, D, cv=splits)

        return reg_preds, prop_preds

    # Calculates DR outcomes
    def calculate_dr_outcomes(
        self,
        D: np.array,
        y: np.array,
        reg_preds,
        prop_preds
    ) -> np.array:

        if self.n_treat == 1:  # if treatment is binary
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
            y_dr_0 = reg_preds[:, 0] + (d0_mask / prop_preds[:, 0]) * (y - reg_preds[:, 0])
            for k in D.unique():  # pick a treatment status
                if k > 0:  # make sure it is not control
                    dk_mask = np.where(D == k, 1, 0)
                    y_dr_k = reg_preds[:, k] + (dk_mask / prop_preds[:, k]) * (y - reg_preds[:, k])
                    dr_k = y_dr_k - y_dr_0  # this is an n x 1 vector
                    dr_vec.append(dr_k)
            dr = np.stack(dr_vec)  # this is an n x k matrix

        return dr

    def cate_fit_predict(self, reg_cate, train, test, dr):
        if self.n_treat == 1:
            reg_cate_fitted = reg_cate.fit(train, dr)
            cate_preds = reg_cate_fitted.predict(test)
        else:
            cate_preds = []
            for k in range(self.n_treat):  # fit a separate cate model for each treatment status?
                reg_cate_fitted = reg_cate.fit(train, dr[:, k])
                cate_preds.append(reg_cate_fitted.predict(test))

            cate_preds = np.stack(cate_preds)

        return cate_preds

    # Fits CATE in training, predicts in validation
    def fit_cate_train(self, reg_cate, Ztrain, Zval):
        return self.cate_fit_predict(reg_cate, Ztrain, Zval, self.dr_train)

    # CV prediction of CATEs
    def fit_cate_cv(self, reg_cate, Z, D, shuffle=True, random_state=712):
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)

        cate_preds = [_ for i in range(self.n_splits)]

        for train, test in cv.split(Z, D):
            cate_preds[test] = self.cate_fit_predict(
                reg_cate=reg_cate,
                train=Z.iloc[train],
                test=Z.iloc[test],
                dr=self.dr_val[train],
            )

        return np.concatenate(cate_preds)
