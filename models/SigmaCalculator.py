from numba import njit
import numpy as np

#f函数的原函数、一二阶导数
@njit
def Letter_f(x, y, k):
    u = np.complex(x,y)
    return (np.square(u) - u )*np.exp((u-1)*k)

@njit
def Flower_Lhat_tT(x, y, option_prices, log_strikes, log_x):
    """
    return numpy.complex64
    """
    return (1 + np.exp(-log_x) * np.sum(Letter_f(x, y, (log_strikes[:-1] - log_x)) *
                                  option_prices[:-1] *
                                  (log_strikes[1:]-log_strikes[:-1])))

##@njit
# def Flower_Lhat_tTao(x, y, optiondata, forwarddata):
#     l = 1+0j
#     for tenor_index, F in enumerate(forwarddata):
#         l *= Flower_Lhat_tT(x, y, option_prices=optiondata[:,0, tenor_index],
#                        log_strikes=optiondata[:,1, tenor_index],
#                        log_x=F)
#     return l

#Find uhat and calculate sigma theta
@njit
def find_uhat(BSIV, tenor, optiondata, log_forward):
    # k = len(forwarddata)
    ubar = np.sqrt(2 / tenor * np.log(20) / np.square(BSIV))
    u_grid = np.arange(0.1, ubar, 0.1)
    L = np.empty(u_grid.shape, dtype=np.complex64)
    for i, ui in enumerate(u_grid):
      L[i] = Flower_Lhat_tT(0, ui, optiondata[:,0], optiondata[:,1], log_forward)
    # uhat1 based on abs(L(u)): 1st time abs(L(u)) <= 0.2
    uhat1_ind = np.where(np.abs(L) <= 0.2)[0]
    if uhat1_ind.size > 0:
      uhat1 = u_grid[uhat1_ind[0]]
    else:
      uhat1 = ubar
    # uhat2 -- abs(L(u)) attains minimum on [0,ubar]
    uhat2 = u_grid[(np.abs(L) == np.min(np.abs(L)))][0]
    # uhat is the minimum of uhat1 and uhat2
    uhat = min(uhat1, uhat2)
    return uhat

@njit
def CalSigma(uhat, tenor, optiondata, log_forward, multiplier=7, sigma_tenor=21):

    sigma = np.sqrt(-2/tenor/np.square(uhat)*np.log(np.abs(Flower_Lhat_tT(0, uhat, optiondata[:,0], optiondata[:,1], log_forward))))
    # theta = multiplier*sigma*np.sqrt(sigma_tenor/252)
    return sigma


def Sigma(datalist):

    """
    把所有的初始信息都转化为(x,y)组合，然后输入给

    :param theta:
    :return:
    """

    BSIV, tenor, interest, spot, forward, optiondata = datalist
    local_optiondata = optiondata.copy()
    local_optiondata[:,1] = np.log(local_optiondata[:,1])
    log_forward = np.log(forward)
    tenor = tenor/365


    uhat = find_uhat(BSIV, tenor, optiondata, log_forward)
    sigma = CalSigma(uhat, tenor, local_optiondata, log_forward, multiplier=7, sigma_tenor=5)
    return sigma


