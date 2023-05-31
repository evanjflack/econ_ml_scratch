import unittest

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from drtester import DRtester


class TestDRTester(unittest.TestCase):

    def _get_data(self, num_treatments=1):
        np.random.seed(123)

        N = 10000  # number of units
        K = 5  # number of covariates

        # Generate random Xs
        X_mu = np.zeros(5)  # Means of Xs
        # Random covariance matrix of Xs
        X_sig = np.diag(np.random.rand(5))
        X = st.multivariate_normal(X_mu, X_sig).rvs(N)

        # Effect of Xs on outcome
        X_beta = np.random.uniform(0, 5, K)
        # Effect of treatment on outcomes
        D_beta = np.arange(num_treatments + 1)
        # Effect of treatment on outcome conditional on X1
        DX1_beta = np.array([0] * num_treatments + [3])

        # Generate 3 treatments (randomly assigned, equal probability)
        D1 = np.random.uniform(0, 1, N)
        if num_treatments == 1:
            propensity = np.maximum(0.2, np.minimum(0.8, 0.5 + X[:, 0] / 3))
            D = np.random.binomial(n=1, p=propensity)
            # D = np.where(
            #     D1 <= 1 / 2,
            #     0,
            #     1
            # )
        else:
            D = np.where(
                D1 <= 1 / 3,
                0,
                np.where(
                    D1 > 2 / 3,
                    2,
                    1
                )
            )

        D_dum = pd.get_dummies(D)

        # Generate Y (based on X, D, and random noise)
        Y_sig = 1  # Variance of random outcome noise
        Y = X @ X_beta + (D_dum @ D_beta) + X[:, 1] * (D_dum @ DX1_beta) + np.random.normal(0, Y_sig, N)
        Y = Y.to_numpy()

        train_prop = .5
        train_N = np.ceil(train_prop * N)
        ind = np.array(range(N))
        train_ind = np.random.choice(N, int(train_N), replace=False)
        val_ind = ind[~np.isin(ind, train_ind)]

        Xtrain, Dtrain, Ytrain = X[train_ind], D[train_ind], Y[train_ind]
        Xval, Dval, Yval = X[val_ind], D[val_ind], Y[val_ind]

        return Xtrain, Dtrain, Ytrain, Xval, Dval, Yval

    def test_multi(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=2)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier()
        reg_y = GradientBoostingRegressor()
        reg_cate = GradientBoostingRegressor()

        # test the DR outcome difference
        my_dr_tester = DRtester(reg_y, reg_t).fit_nuisance(
            Xval, Dval, Yval, Xtrain, Dtrain, Ytrain
        )
        dr_outcomes = my_dr_tester.dr_val

        ates = dr_outcomes.mean(axis=0)
        for k in range(dr_outcomes.shape[1]):
            ate_errs = np.sqrt(((dr_outcomes[:, k] - ates[k]) ** 2).sum() / \
                      (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))

            assert abs(ates[k] - (k + 1)) < 2 * ate_errs

        Ztrain = Xval[:, 1]
        Zval = Xtrain[:, 1]

        my_dr_tester.fit_cate(reg_cate, Ztrain, Zval)

        my_dr_tester = my_dr_tester.evaluate_all()

        assert my_dr_tester.df_res.blp_pval.values[0] > 0.1  # no heterogeneity
        assert my_dr_tester.df_res.blp_pval.values[1] < 0.05  # heterogeneity

        assert my_dr_tester.df_res.cal_r_squared.values[0] < 0.2  # poor R2
        assert my_dr_tester.df_res.cal_r_squared.values[1] > 0.5  # good R2

        assert my_dr_tester.df_res.qini_pval.values[0] > 0.1  # no heterogeneity
        assert my_dr_tester.df_res.qini_pval.values[1] < 0.05  # heterogeneity

    def test_binary(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=1)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier()
        reg_y = GradientBoostingRegressor()
        reg_cate = GradientBoostingRegressor()

        # test the DR outcome difference
        my_dr_tester = DRtester(reg_y, reg_t).fit_nuisance(
            Xval, Dval, Yval, Xtrain, Dtrain, Ytrain
        )
        dr_outcomes = my_dr_tester.dr_val

        ate = dr_outcomes.mean(axis=0)
        ate_err = np.sqrt(((dr_outcomes - ate) ** 2).sum() / \
                               (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))
        truth = 1
        assert abs(ate - truth) < 2 * ate_err

        Ztrain = Xval[:, 1]
        Zval = Xtrain[:, 1]

        my_dr_tester.fit_cate(reg_cate, Ztrain, Zval)

        my_dr_tester = my_dr_tester.evaluate_all()

        assert my_dr_tester.df_res.blp_pval.values[0] < 0.05  # heterogeneity
        assert my_dr_tester.df_res.cal_r_squared.values[0] > 0.5  # good R2
        assert my_dr_tester.df_res.qini_pval.values[0] < 0.05  # heterogeneity
