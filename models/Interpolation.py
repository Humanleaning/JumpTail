"""
接到一个列表，返回一个列表，包含格式化的
"""
import numpy as np
from numba import njit


##@njit searchsorted 有bug
def InterpolationData(datalist):
    newlist = []
    for item in datalist:
        tenor, interest, spot, forward, data = item
        index = np.searchsorted(data[:,1], forward[0], side='left')
        BSIV = data[index,0]

        left = np.exp(np.log(spot)-8*BSIV*np.sqrt(tenor/365))
        right = np.exp(np.log(spot)+8*BSIV*np.sqrt(tenor/365))
        strike_grid = np.linspace(left, right, 300)
        iv_grid = np.interp(strike_grid, data[:,1], data[:,0])
        data = np.hstack((iv_grid, strike_grid))
        newlist.append([BSIV,tenor,interest,spot,forward,data])
    return newlist


from scipy.stats import norm
def Blacks_P(K, tenor, F, r, iv):  # 日期都是以日为单位，需要转化
    tenor = tenor / 365
    d1 = ((np.log(F / K) + np.square(iv) * tenor / 2)) / (iv * np.sqrt(tenor))
    d2 = d1 - iv * np.sqrt(tenor)
    price = np.exp(-r * tenor) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return price
def Blacks_C(K, tenor, F, r, iv):  # 日期都是以日为单位，需要转化
    tenor = tenor / 365
    d1 = ((np.log(F / K) + np.square(iv) * tenor / 2)) / (iv * np.sqrt(tenor))
    d2 = d1 - iv * np.sqrt(tenor)
    price = np.exp(-r * tenor) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return price
def Convert2price(datalist):
    newlist = []
    for item in datalist:
        BSIV, tenor, interest, spot, forward, data = item
        strike_grid = data[:,1]
        iv_grid = data[:,0]
        indexp = np.where(strike_grid<=forward)[0]
        indexc = np.where(strike_grid> forward)[0]

        price_gridp  = Blacks_P(strike_grid[indexp], tenor, forward, interest, iv_grid[indexp])
        price_gridc = Blacks_C(strike_grid[indexc], tenor, forward, interest, iv_grid[indexc])
        price_grid = np.hstack((price_gridp, price_gridc))
        data = np.vstack((price_grid, strike_grid)).T
        newlist.append([BSIV, tenor, interest, spot, forward, data])
    #补丁，因为原先的程序写的是三维索引，所以如果是一维的话再增加一维

    return newlist