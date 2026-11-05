"""Backtest со скользящим origin. Прогноз суммарного net-спроса на H месяцев вперёд, данные строго < origin.
Модели: model (сезонность×уровень) и три baseline. Метрики: WAPE, bias, + stock-free proxy политики
(target = F + z·σ·√H: доля покрытого спроса = fill, избыток = excess)."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from .forecast import Prepared

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_backtest(raw, cfg, cur, horizons=(3, 6), first_origin="2023-01", step=1, censor=True, halflife=None):
    fc = cfg["forecast"]
    hl = halflife or fc["halflife_months"]
    P = Prepared(raw, cur, fc["censor_from"], censor=censor)
    T = len(P.months)
    z = norm.ppf(cfg["service_level"])
    j0 = list(P.months).index(pd.Period(first_origin, "M"))
    rows = []
    for H in horizons:
        for j in range(j0, T - H + 1, step):
            idx, rows_idx = P.seasonal_by_row(j, fc["seasonal_min_months"])
            f = P.fit(j, rows_idx, fc["window_months"], hl, fc["min_obs"])
            moy_f = P.moy[j:j + H]
            seas_f = rows_idx[:, moy_f].sum(1)
            actual = np.nan_to_num(P.M[:, j:j + H]).sum(1)
            prev12 = P.M[:, max(0, j - 12):j]
            active = np.nan_to_num(prev12).sum(1) > 0
            m_model = f["level"] * seas_f
            m_t12 = np.nan_to_num(np.nanmean(np.where(np.isnan(prev12), np.nan, prev12), axis=1)) * H if prev12.shape[1] else 0
            sn = np.nan_to_num(P.M[:, [k - 12 for k in range(j, j + H)] if j - 12 >= 0 else []]).sum(1) if j >= 12 else np.zeros(len(P.M))
            ma3 = np.nan_to_num(np.nanmean(P.M[:, j - 3:j], axis=1)) * H
            sigma = f["sigma"] * np.sqrt(seas_f)
            m_blend = np.where(sn > 0, 0.5 * m_model + 0.5 * sn, m_model)
            for name, F in (("model", m_model), ("blend", m_blend), ("trailing12", m_t12), ("seasonal_naive", sn), ("ma3", ma3)):
                T_ = F + z * sigma
                for i in np.where(active)[0]:
                    rows.append((H, str(P.months[j]), P.cat[i], name, F[i], actual[i], T_[i]))
    df = pd.DataFrame(rows, columns=["H", "origin", "category", "model", "fcst", "actual", "target"])
    return df


def summarize(df):
    def agg(g):
        a = g["actual"].sum()
        short = np.maximum(g["actual"] - g["target"], 0).sum()
        exc = np.maximum(g["target"] - g["actual"], 0).sum()
        return pd.Series({"n": len(g), "wape": (g["fcst"] - g["actual"]).abs().sum() / a if a else np.nan,
                          "bias": (g["fcst"].sum() - a) / a if a else np.nan,
                          "fill_proxy": 1 - short / a if a else np.nan,
                          "excess_ratio": exc / g["target"].sum() if g["target"].sum() else np.nan})
    overall = df.groupby(["H", "model"]).apply(agg, include_groups=False).reset_index()
    bycat = df.groupby(["H", "category", "model"]).apply(agg, include_groups=False).reset_index()
    return overall, bycat
