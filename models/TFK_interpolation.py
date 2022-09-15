import numpy
import numpy as np
from scipy import optimize
from statsmodels.nonparametric._kernel_base import LeaveOneOut
from scipy.special import logsumexp
from scipy.stats import norm

class OM_KernelReg(object):
    """
    用于生成隐含波动率，首先输入训练集y=[Sigma]，X=[Vega，XYZ]，(array-like)
    x=logT y=delta，z=1or0,其中，
        T是以日为单位，
        delta是call_equivalent_delta,而且包含收益折现因子
        z当看涨期权是1，看跌期权是0
    bw控制超参数优化方法,这里设定原则是只能比OptionMetrics给的参数更平滑(int)
    最后使用.fit()来输出data_predict=Xi所在位置的iv=yi(array-like)
    """

    def __init__(self, Sigma, Vega, XYZ, bw_optimal_method=2):
        """

        :param Sigma: 期权隐含波动率
        :param Vega: 期权隐含vega
        :param XYZ: x=logT y=delta，z=1or0,其中，T是以日为单位，delta是call_equivalent_delta,而且包含收益折现因子
        :param bw_optimal_method:
        """
        self.endog_Sigma = np.reshape(np.asarray(Sigma), (-1, 1))
        self.exog_Vega = np.reshape(np.asarray(Vega), (-1, 1))
        self.exog_XYZ = np.reshape(np.asarray(XYZ), (-1, 3))
        # self.exog = np.column_stack((self.Vega,self.XYZ))
        # self.k_vars = np.shape(self.Predictor)[1]
        self.nobs = np.shape(self.exog_Vega)[0]
        self.est = self._est_optionmetrics
        self.bw = self._compute_reg_bw(bw_optimal_method)

    def _compute_reg_bw(self, bw):
        """
        用来设定整个模型的bandwidth
        :param bw:
        :return:
        """
        # 传统意义上bandwidth value should be se to (0,+inf)
        # this bandwidth is square of bandwidth，所以不可以是负的
        # X = np.std(self.exog, axis=0)
        # h0 = 1.06 * X * \self.nobs ** (- 1. / (4 + np.size(self.exog, axis=1)))
        if bw == 1:  # 使用OptionMetrics给定的
            self._bw_method = "OptionMetrics-specified"
            func = self.est
            bw = np.array([0.05, 0.005, 0.001])
            self.MSE = self.cv_loo(bw, func)
            return bw
        elif bw == 2:  # 使用NM算法进行优化，速度比较快
            self._bw_method = 'Nelder-Mead'
            res = self.cv_loo
            func = self.est
            h0 = np.array([0.05, 0.005, 0.001])
            bounds = [(0.05, 1), (0.005, 1), (0.001, 1)]
            optimal_results = optimize.minimize(res, x0=h0, args=(func,), method='Nelder-Mead', bounds=bounds)
            bw_estimated = optimal_results.x
            self.MSE = optimal_results.fun
            return bw_estimated
        elif bw == 3:  # 使用遗传算法进行优化，速度比较慢，但结果并没有更好
            self._bw_method = 'differential_evolution'
            res = self.cv_loo
            func = self.est
            bounds = [(0.05, 1), (0.005, 1), (0.001, 1)]
            optimal_results = optimize.differential_evolution(res, bounds, args=(func,))
            bw_estimated = optimal_results.x
            self.MSE = optimal_results.fun
            return bw_estimated
        elif bw == 4:  # 使用Powell，速度比较慢，但结果并没有更好
            self._bw_method = 'Powell'
            res = self.cv_loo
            func = self.est
            h0 = np.array([0.05, 0.005, 0.001])
            bounds = [(0.05, 1), (0.005, 1), (0.001, 1)]
            optimal_results = optimize.minimize(res, x0=h0, args=(func,), method='Powell', bounds=bounds)
            bw_estimated = optimal_results.x
            self.MSE = optimal_results.fun
            return bw_estimated
        elif bw == 5:
            self._bw_method = 'fmin'
            res = self.cv_loo
            func = self.est
            h0 = np.array([0.05, 0.005, 0.001])
            optimal_results = optimize.fmin(res, x0=h0, args=(func,), maxiter=1e3, maxfun=1e3, disp=0, full_output=True)
            bw_estimated = optimal_results[0]
            self.MSE = optimal_results[1]
            return bw_estimated
        else:
            self._bw_method = "Error_setto_OptionMetrics-specified"
            return np.array([0.05, 0.005, 0.001])

    def _est_optionmetrics(self, bw, endog_Sigma, exog_Vega, exog_XYZ, data_predict):
        """
        OptionMetrics kernel regression

        Parameters
        ----------
        bw : array_like
            Array of bandwidth value(s).
        endog_Sigma : 2D array_like(n*1)
            The dependent variable.
        exog_Vega  : 2D array_like(n*1)
            The independent variable(s).
        exog_XYZ    : 2D array_like(n*3)
            The independent variable(s).
        data_predict: 2D array_like(1*3)
            The point(s) at which the density is estimated.

        Returns
        -------
        sgima : ndarray
            The value of the conditional mean at `data_predict`.
        """
        # data_predict = np.reshape(data_predict,(1,3))
        # bw_muti = np.array([-10, -100, -500]).reshape(1,3)#OptionMetrics bandwidth
        # print(bw)
        bw_muti = (-(1 / 2) * (1 / bw)).reshape(1, 3)
        distance = np.sum(np.square(exog_XYZ - data_predict) * bw_muti, axis=1)
        endog_Sigma = endog_Sigma.T
        exog_Vega = exog_Vega.T
        sigma = np.exp(logsumexp(distance, b=endog_Sigma * exog_Vega) - logsumexp(distance, b=exog_Vega))
        return sigma

    def cv_loo(self, bw, func):
        r"""
        优化算法的底层函数
        The cross-validation function with leave-one-out estimator.

        Parameters
        ----------
        bw : array_like
            Vector of bandwidth values.
        func : callable function
            Returns the estimator of g(x).  Can be either ``_est_loc_constant``
            (local constant) or ``_est_loc_linear`` (local_linear).

        Returns
        -------
        L : float
            The value of the CV function.

        Notes
        -----
        Calculates the cross-validation least-squares function. This function
        is minimized by compute_bw to calculate the optimal value of `bw`.

        For details see p.35 in [2]

        .. math:: CV(h)=n^{-1}\sum_{i=1}^{n}(Y_{i}-g_{-i}(X_{i}))^{2}

        where :math:`g_{-i}(X_{i})` is the leave-one-out estimator of g(X)
        and :math:`h` is the vector of bandwidths
        """
        LOO_Sigma = LeaveOneOut(self.endog_Sigma).__iter__()
        LOO_Vega = LeaveOneOut(self.exog_Vega).__iter__()
        LOO_XYZ = LeaveOneOut(self.exog_XYZ)

        L = 0
        for ii, XYZ_not_i in enumerate(LOO_XYZ):
            Sigma_not_i = next(LOO_Sigma)
            Vega_not_i = next(LOO_Vega)
            G = func(bw,
                     endog_Sigma=Sigma_not_i,
                     exog_Vega=Vega_not_i,
                     exog_XYZ=XYZ_not_i,
                     data_predict=self.exog_XYZ[ii, :])
            L += (self.endog_Sigma[ii] - G) ** 2

        # Note: There might be a way to vectorize this. See p.72 in [1]
        return L / self.nobs

    def fit_sigma(self, regressor=np.array([1, 2, 3])):
        """
        输出函数
        """
        func = self.est
        regressor = np.reshape(regressor, (1, 3))
        mean_sigma = func(self.bw, self.endog_Sigma, self.exog_Vega, self.exog_XYZ,
                          data_predict=regressor)
        return mean_sigma

    def fit_sigma_k(self, regressor, marketinfo):
        F, S, r = marketinfo
        logT, ce_delta, call_put = regressor
        T = np.exp(logT)/365
        sigma = self.fit_sigma(regressor)
        if call_put:  # 看涨期权
            multi = F / (S * np.exp(r*T))
            nd1 = ce_delta / multi
            K = F / np.exp(norm.ppf(nd1)*sigma*np.sqrt(T) - np.square(sigma)/2*T)
        else:  # 看跌期权
            multi = F / (S * np.exp(r*T))
            nd1 = (ce_delta-1) / multi + 1
            K = F / np.exp(norm.ppf(nd1)*sigma*np.sqrt(T) - np.square(sigma)/2*T)
        return sigma, K



