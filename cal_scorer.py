class cal_scorer:
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
        n_groups
    ):
        self.cate_model = cate_model
        self.model_y_zero = model_y_zero
        self.model_y_one = model_y_one
        self.model_t = model_t
        self.n_groups = n_groups

    def calculate_dr_outcomes(
        self,
        X,
        D,
        y
    ):
        reg_zero_preds= self.model_y_zero.predict(X)
        reg_one_preds = self.model_y_one.predict(X)
        reg_preds = reg_zero_preds * (1 - D) + reg_one_preds * D
        prop_preds = self.model_t.predict(X)

        dr = reg_one_preds - reg_zero_preds
        reisz = (D - prop_preds) / np.clip(prop_preds * (1 - prop_preds), .01, np.inf)
        dr += (y - reg_preds) * reisz

        return dr

    def score(
        self,
        Xval,
        Dval,
        Yval,
        Zval,
        Ztest
    ):

        self.dr_outcomes_val_ = self.calculate_dr_outcomes(Xval, Dval, Yval)
        self.cate_preds_val_ = self.cate_model.predict(Zval)
        self.cate_preds_test_ = self.cate_model.predict(Ztest)

        # Define quantiles, based on test set
        cuts = np.quantile(self.cate_preds_test_, np.linspace(0, 1, self.n_groups + 1))

        # Calculate GATE and average CATE prediction for each group (in validation set)
        self.probs = np.zeros(self.n_groups)
        self.g_cate = np.zeros(self.n_groups)
        self.gate = np.zeros(self.n_groups)
        for i in range(self.n_groups):
            ind = (self.cate_preds_val_ >= cuts[i]) & (self.cate_preds_val_ < cuts[i + 1])
            self.probs[i] = np.mean(ind)
            self.gate[i] = np.mean(self.dr_outcomes_val_[ind])
            self.g_cate[i] = np.mean(self.cate_preds_val_[ind])

        # Calculated overall ATE
        ate = np.mean(self.dr_outcomes_val_)

        # Calculate calibration score
        diff1 = np.sum(abs(gate - g_cate) * probs)
        diff2 = np.sum(abs(gate - ate) * probs)
        self.cal_score = 1 - (diff1 / diff2)

        return self