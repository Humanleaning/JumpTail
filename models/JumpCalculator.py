"""

输入sigma,期权数据


"""
from  numba import njit
import numpy as np
from scipy.optimize import least_squares

@njit
def calc_exponentiated_mean(beta, x):
    """
    计算exp mean
    :param beta:
    :param x:
    :return:
    """
    lin_combi = np.asarray(beta) @ np.asarray(x)
    mean = np.exp(lin_combi)
    return mean

@njit
def calc_residual(beta, x, y_obs):
    """
    计算residual
    :param beta:
    :param x:
    :param y_obs:
    :return:
    """
    y_pred = calc_exponentiated_mean(beta, x)
    r = (y_pred - y_obs).flatten()
    return r


def LV_T( X, y):
    """
    输入参数theta和等待回归的X,y，输出LV
    :param theta:
    :param X:
    :param y:
    :return:
    """
    result_nls_lm = least_squares(fun=calc_residual, x0=(1,1), args=(X, y), method='lm')
    b0, b1 = result_nls_lm.x
    alpha = b1-1
    phi = np.exp(b0)*alpha*(alpha+1)
    #LV = np.exp(-1*theta*alpha) * phi * (alpha*theta*(alpha*theta+2)+2) / np.power(alpha,3)
    return alpha, phi
def LV_Tao(datalist ):
    """
    把所有的初始信息都转化为(x,y)组合，然后输入给

    :param theta:
    :return:
    """


    BSIV, tenor, interest, spot, forward, data = datalist
    tenor = tenor/365
    thresh = np.exp(np.log(spot)-2*BSIV*np.sqrt(tenor/365))
    index = np.where(data[:,1]<thresh)[0]

    y = data[index,0] * np.exp(interest) / (forward*tenor)
    X_ = np.log(data[index, 1]) - np.log(forward)
    X = np.vstack((np.ones(X_.shape), X_))


    alpha, phi = LV_T(X, y)
    return alpha, phi


def RV_T(X, y):
    """
    输入参数theta和等待回归的X,y，输出LV
    :param theta:
    :param X:
    :param y:
    :return:
    """
    result_nls_lm = least_squares(fun=calc_residual, x0=(1,1), args=(X, y), method='lm')
    b0, b1 = result_nls_lm.x
    alpha = -b1+1
    phi = np.exp(b0)*alpha*(alpha-1)
    #LV = np.exp(-1*theta*alpha) * phi * (alpha*theta*(alpha*theta+2)+2) / np.power(alpha,3)
    return alpha, phi
def RV_Tao(datalist ):
    """
    把所有的初始信息都转化为(x,y)组合，然后输入给

    :param theta:
    :return:
    """


    BSIV, tenor, interest, spot, forward, data = datalist
    tenor = tenor/365
    thresh = np.exp(np.log(spot)+1*BSIV*np.sqrt(tenor/365))
    index = np.where(data[:,1]>thresh)[0]

    y = data[index,0] * np.exp(interest) / (forward*tenor)
    X_ = np.log(data[index, 1]) - np.log(forward)
    X = np.vstack((np.ones(X_.shape), X_))


    alpha, phi = RV_T(X, y)
    return alpha, phi