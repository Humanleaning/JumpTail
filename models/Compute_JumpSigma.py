




from models.SigmaCalculator import Sigma
from models.Interpolation import InterpolationData, Convert2price
from models.JumpCalculator import LV_Tao, RV_Tao
import copy


def compute_JS(info_ex, date, df):
    # try:
    info = info_ex.getoneday(date)
    # except:
    #     for alpha in alphas:
    #         VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = 0.1
    #     return VaRandVol

    # try:
    sigma = Sigma(info)
    alpha_l, phi_l = LV_Tao(info)
    alpha_r, phi_r = RV_Tao(info)
    # except:
    #     sigma = -0.01
    #     alpha = -0.01
    #     phi = -0.01
    #     LV = -0.01

    df.loc[date, "sigma"] = sigma
    df.loc[date, 'alpha_l'] = alpha_l
    df.loc[date, 'phi_l'] = phi_l
    df.loc[date, 'alpha_r'] = alpha_r
    df.loc[date, 'phi_r'] = phi_r

    return df