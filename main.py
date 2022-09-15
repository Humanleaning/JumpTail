#1.是否统一了一套变量命名体系
#2.是否进行了数据筛选
    # 2.1 是否需要删除边缘tenor
    # 2.2 是否需要删除边缘strike
    # 2.3 是否需要删除周期权
    # 2.4 是否需要删除其他size的期权
    # 2.5 是否需要矫正tenor
#3.是否进行了新变量的生成


import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from models.InfoExtractor import IvyDBUS_InfoExtractor, IvyDBEurope_InfoExtractor
from models.Compute_JumpSigma import compute_JS


IvyDB_European_list = ['France','German','Italy','Netherlands','UK']
data_url = r'data/'

markets = ['Brazil', 'Italy', 'China', 'France', 'German', 'Korea', 'Mexico', 'Netherlands', 'Oil', 'Russia', 'UK', 'US']
markets = ['GLD', 'SLV', 'Oil', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
markets = ['XLU', 'XLV', 'XLY']
#%%
for market in tqdm(markets):
# market = 'XLC'
    if market in IvyDB_European_list:
        with open(data_url + market + '/Style1.pkl', 'rb') as file:
            Options, Spots, Interests, Dividend, Forwards = pickle.load(file)
        info_ex = IvyDBEurope_InfoExtractor(market, Options, Forwards, Interests, Spots, targe_tenor=30)
    else:
        with open(data_url + market + '/Style1.pkl', 'rb') as file:
            Options, Spot, Forward, Interest = pickle.load(file)
        info_ex = IvyDBUS_InfoExtractor(Options, Forward, Interest, Spot, market, targe_tenor=30)

    df = pd.DataFrame()  # 构建一个数据框，用来存储数据
    Dates = Options.index.unique()
    # Dates = Dates[0:2000]
    # Dates = Dates[Dates >= pd.to_datetime('2018-12-9')]
    for date in tqdm(Dates):
        df = compute_JS(info_ex, date, df)



    for i in range(8,11):
        theta = i * df.sigma * np.sqrt(5 / 252)
        df['LJVariance'+str(i)] = np.exp(-1 * theta * df.alpha_l) * df.phi_l * (df.alpha_l * theta * (df.alpha_l * theta + 2) + 2) / np.power(df.alpha_l, 3)
        df['LJVolatility'+str(i)] = np.sqrt(df['LJVariance'+str(i)])

        df['RJVariance' + str(i)] = np.exp(-1 * theta * df.alpha_r) * df.phi_r * (df.alpha_r * theta * (df.alpha_r * theta + 2) + 2) / np.power(df.alpha_r, 3)
        df['RJVolatility' + str(i)] = np.sqrt(df['RJVariance' + str(i)])

    df.to_csv(data_url + market + '/result_new.csv')

#%%

#%%


# df.to_csv(market+'data.csv')



#%%
# for market in tqdm(IvyDB_European_list):
#     with open(data_url + market + '/Style1.pkl', 'rb') as file:
#         Options, Spots, Interests, Dividend, Forwards = pickle.load(file)
#     Options = pd.read_csv(data_url + market + '/options.csv', parse_dates=['Date', 'Expiration'])
#     Options.rename(columns={'ImpliedVolatility': 'impl_volatility',
#                             'Delta': 'delta',
#                             'Date': 'date',
#                             'Vega':'vega',
#                             'Strike': 'strike_price',
#                             'CallPut': 'cp_flag',
#                             'Expiration': 'exdate',
#                             'Bid': 'best_bid',
#                             'Ask': 'best_offer'}, inplace=True)
#     Options['strike_price'] = Options.strike_price/1000
#     Options.set_index('date', inplace=True)
#     Options = pd.merge(Options, Spots, how='left', left_index=True, right_index=True)
#     Options['moneynessS'] = Options.strike_price / Options.spot
#     Options.reset_index(inplace=True)
#     Options['tenor'] = (Options.exdate-Options.date).dt.days
#     Options.set_index('date', inplace=True)
#     with open(data_url + market + '/Style1.pkl', 'wb') as file:
#         pickle.dump((Options, Spots, Interests, Dividend, Forwards), file)

