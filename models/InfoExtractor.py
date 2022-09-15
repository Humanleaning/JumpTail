"""
目标功能，从原始数据中提取能够用于计算JumpVariation的数据

# 输入所有需要的数据集, 然后进行数据清洗
# 对清洗后的数据进行规整，并生成插值对象
# 插值出一个30天的call和put列，put就[0.47, 1] call就[0, 5.2]
# 对插值结果进行矫正，只留下OTM的部分
# 对插值结果计算期权价格
# 装箱送出
"""

# 只需要给定一个
from scipy.stats import norm
import numpy as np
from models.TFK_interpolation import OM_KernelReg


def Get_r(interestset, date, tenor):
    Interest = interestset.loc[date]
    # if np.all(np.diff(Interest.tenor) > 0):
    return np.interp(tenor, Interest.tenor.values, Interest.interest.values)
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


# %%
# 用于提取IvyDB US的数据
class IvyDBUS_InfoExtractor(object):

    def __init__(self, Options, Forward, Interest, Spot, market, targe_tenor=30):
        # 2.1 是否需要删除边缘tenor
        # 2.2 是否需要删除波动率异常（违反无套利）的期权
        # 2.3 是否需要删除边缘strike（best bid过小的）
        # 2.3 是否需要删除周期权
        # 2.4 是否需要删除其他size的期权
        # 2.5 是否需要矫正tenor
        # 2.6 是否只留下OTM期权


        # 2.1只留下tenor在[8-42]的期权
        # Options = Options.loc[(Options.tenor >=8) & (Options.tenor <=45)]

        # 2.2 是否需要删除波动率异常（违反无套利）的期权
        Options = Options.loc[Options.impl_volatility > 0]

        # 2.3 去掉askbidspread过大的
        Options['ab_spread'] = (Options.best_offer - Options.best_bid)
        Options = Options.loc[Options.ab_spread < 5 * Options.best_bid]

        # 2.4 去掉midprice相同的, 保留距离中值最小的那个
        Options['mid_price'] = (Options.best_bid + Options.best_offer) / 2
        Options['distance_to_ATM'] = np.fabs(Options.moneynessS - 1)
        Options.reset_index(inplace=True)
        Options = Options.sort_values(['date', 'cp_flag', 'tenor', 'distance_to_ATM'], ascending=True)
        Options.drop_duplicates(subset=['date', 'tenor', 'cp_flag', 'mid_price'], keep='first', inplace=True, ignore_index=True)  # 保留距离中值最小的那个
        Options.set_index('date', inplace=True)

        # 2.5 取消best bid过小的
        Options = Options.loc[Options.best_bid >= 0]

        # 2.6 是否只留下OTM期权
        # OTM_Options = Options.loc[((Options.cp_flag == 'C') & (Options.moneynessF >= 1)) |
        #                           ((Options.cp_flag == 'P') & (Options.moneynessF <= 1))]


        # 将期权排序
        # OTM_Options.reset_index(inplace=True)
        # OTM_Options = OTM_Options.sort_values(['date', 'tenor', 'strike_price'], ascending=True)
        # OTM_Options.set_index('date', inplace=True)

        Options['logT'] = np.log(Options.tenor)
        Options['CEdelta'] = np.where(Options.cp_flag == 'C', Options.delta, Options.delta + 1)
        Options['type'] = Options.cp_flag.map({'C': 1, 'P': 0})


        self.Options = Options
        self.Forward = Forward
        self.Interest = Interest
        self.Spot = Spot
        self.target_tenor = targe_tenor
        self.logT = np.log(self.target_tenor)

    def getoneday(self, date):
        Option = self.Options.loc[date]
        r = Get_r(self.Interest, date, self.target_tenor)
        F = self.Forward.loc[(date, self.target_tenor), 'forward_price']
        try:
            S = self.Spot.loc[date, 'spot']
        except:
            S = F/np.exp(r*self.target_tenor/365)




        num = 250
        marketinfo = np.array([F,S,r])
        omkr = OM_KernelReg(Option.impl_volatility, Option.vega, Option[['logT', 'CEdelta', 'type']], bw_optimal_method=1)
        call_delta_grid1 = np.exp(np.linspace(np.log(0.00001),np.log(0.25), num=int(2/3*num), endpoint=False))
        call_delta_grid2 = np.linspace(0.25, 0.52, num=int(1 / 3 * num), endpoint=True)
        call_delta_grid = np.hstack((call_delta_grid1, call_delta_grid2))
        call_len = len(call_delta_grid)
        call_grid = np.zeros((call_len,2))


        put_delta_grid1 = np.linspace(0.47, 0.8, num=int(1 / 3 * num), endpoint=False)
        put_delta_grid2 = np.exp(np.linspace(np.log(0.8),np.log(0.9999), num=int(2/3*num), endpoint=True))
        put_delta_grid3 = np.arange(0.99991, 0.999999, 0.000002)
        put_delta_grid = np.hstack((put_delta_grid1, put_delta_grid2, put_delta_grid3))
        put_len = len(put_delta_grid)
        put_grid = np.zeros((put_len, 2))

        for i, ce_delta in enumerate(call_delta_grid):
            call_grid[call_len-1-i] = omkr.fit_sigma_k(np.array([self.logT, ce_delta, 1]), marketinfo)
        call_grid = call_grid[call_grid[:, 1] > F, :]
        call_grid[:,0] = Blacks_C(call_grid[:, 1], self.target_tenor, F, r, call_grid[:, 0])
        for i, ce_delta in enumerate(put_delta_grid):
            put_grid[put_len-1-i] = omkr.fit_sigma_k(np.array([self.logT, ce_delta, 0]), marketinfo)
        put_grid = put_grid[put_grid[:, 1] < F, :]
        BSIV = put_grid[-1, 0]
        put_grid[:, 0] = Blacks_P(put_grid[:, 1], self.target_tenor, F, r, put_grid[:, 0])


        optiondata = np.vstack((put_grid, call_grid))

        return [BSIV, self.target_tenor, r, S, F, optiondata]

#%%
# 用于提取IvyDB European的数据
class IvyDBEurope_InfoExtractor(object):

    def __init__(self, market, Options, Forwards, Interests, Spots, targe_tenor=30):


        # # 只留下tenor在[8-42]的期权
        # Options = Options.loc[(Options.tenor >= 8) & (Options.tenor <= 45)]

        # 去除iv小于0期权
        Options = Options.loc[Options.impl_volatility>0]
        Options = Options.loc[Options.tenor>0]

        #
        Options = Options.loc[(Options.CalculationPrice != 'A') & (Options.CalculationPrice != 'B')]


        # 去掉best_bid过小的
        minbids = {'Netherlands': 0.1, 'France': 0.1, 'German': 0.1, 'Italy': 0.1, 'UK': 0.1}
        minbid = minbids[market]
        Options = Options.loc[Options.Last >= minbid]

        # 去掉midprice相同的
        # Options['mid_price'] = Options.Last
        # Options['distance_to_ATM'] = np.fabs(Options.moneynessS - 1)
        # Options.reset_index(inplace=True)
        # Options = Options.sort_values(['date', 'cp_flag', 'tenor', 'distance_to_ATM'], ascending=True)
        # Options.drop_duplicates(subset=['date', 'tenor', 'cp_flag', 'mid_price'], keep='first', inplace=True, ignore_index=True)  # 保留距离中值最小的那个
        # Options.set_index('date', inplace=True)

        # 去掉askbidspread过大的
        # multipliers = {'Netherlands': 10, 'France': 10, 'German': 10, 'Italy': 10, 'UK': 10}
        # multiplier = multipliers[market]
        # Options['ab_spread'] = (Options.best_offer - Options.best_bid)
        # Options = Options.loc[Options.ab_spread < multiplier * Options.best_bid]



        # OTM_Options = Options.loc[((Options.cp_flag == 'C') & (Options.moneynessF >= 1)) |
        #                           ((Options.cp_flag == 'P') & (Options.moneynessF <= 1))]


        # OTM_Options.reset_index(inplace=True)
        # OTM_Options = OTM_Options.sort_values(['date', 'tenor', 'strike_price'], ascending=True)
        # OTM_Options.set_index('date', inplace=True)
        Options['logT'] = np.log(Options.tenor)
        Options['CEdelta'] = np.where(Options.cp_flag == 'C', Options.delta, Options.delta + 1)
        Options['type'] = Options.cp_flag.map({'C': 1, 'P': 0})

        self.Options = Options
        self.Forward = Forwards
        self.Interest = Interests
        self.Spot = Spots
        self.target_tenor = targe_tenor
        self.logT = np.log(self.target_tenor)


    def getoneday(self, date):
        Option = self.Options.loc[date]
        r = Get_r(self.Interest, date, self.target_tenor)
        F = self.Forward.loc[(date, self.target_tenor), 'forward_price']
        try:
            S = self.Spot.loc[date, 'spot']
        except:
            S = F/np.exp(r*self.target_tenor/365)




        num = 250
        marketinfo = np.array([F,S,r])
        omkr = OM_KernelReg(Option.impl_volatility, Option.vega, Option[['logT', 'CEdelta', 'type']], bw_optimal_method=1)
        call_delta_grid1 = np.exp(np.linspace(np.log(0.00001),np.log(0.25), num=int(2/3*num), endpoint=False))
        call_delta_grid2 = np.linspace(0.25, 0.52, num=int(1 / 3 * num), endpoint=True)
        call_delta_grid = np.hstack((call_delta_grid1, call_delta_grid2))
        call_len = len(call_delta_grid)
        call_grid = np.zeros((call_len,2))


        put_delta_grid1 = np.linspace(0.47, 0.8, num=int(1 / 3 * num), endpoint=False)
        put_delta_grid2 = np.exp(np.linspace(np.log(0.8),np.log(0.9999), num=int(2/3*num), endpoint=True))
        put_delta_grid3 = np.arange(0.99991, 0.999999, 0.000002)
        put_delta_grid = np.hstack((put_delta_grid1, put_delta_grid2, put_delta_grid3))
        put_len = len(put_delta_grid)
        put_grid = np.zeros((put_len, 2))

        for i, ce_delta in enumerate(call_delta_grid):
            call_grid[call_len-1-i] = omkr.fit_sigma_k(np.array([self.logT, ce_delta, 1]), marketinfo)
        call_grid = call_grid[call_grid[:, 1] > F, :]
        call_grid[:,0] = Blacks_C(call_grid[:, 1], self.target_tenor, F, r, call_grid[:, 0])
        for i, ce_delta in enumerate(put_delta_grid):
            put_grid[put_len-1-i] = omkr.fit_sigma_k(np.array([self.logT, ce_delta, 0]), marketinfo)
        put_grid = put_grid[put_grid[:, 1] < F, :]
        BSIV = put_grid[-1, 0]
        put_grid[:, 0] = Blacks_P(put_grid[:, 1], self.target_tenor, F, r, put_grid[:, 0])


        optiondata = np.vstack((put_grid, call_grid))

        return [BSIV, self.target_tenor, r, S, F, optiondata]

