"""Прогноз net-спроса: категорийный сезонный индекс × уровень SKU (экспоненциально взвешенный,
десезонированный, с исключением цензурированных месяцев)."""
import numpy as np
import pandas as pd


def seasonal_index(total, min_months=24):
    """Индекс по месяцам года (янв=0..дек=11), среднее=1.
    Отношение ряда к центрированной скользящей 2x12, медиана по годам. total: pd.Series с PeriodIndex."""
    s = total.astype(float)
    if len(s) < min_months:
        return np.ones(12)
    ma = s.rolling(12).mean().rolling(2).mean().shift(-6)
    ratio = (s / ma).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return np.ones(12)
    moy = pd.Index([p.month for p in ratio.index])
    idx = pd.Series(ratio.values, index=moy).groupby(level=0).median().reindex(range(1, 13)).fillna(1.0)
    return (idx / idx.mean()).values


class Prepared:
    """Матрицы по полным месяцам < cur (неполный месяц сюда НЕ попадает)."""

    def __init__(self, raw, cur, censor_from="2024-01", censor=True):
        self.cur = cur
        self.months = pd.period_range("2020-11", cur - 1, freq="M")
        self.sku_ids = raw["sku"]["sku_id"].to_numpy()
        self.cat = raw["sku"]["category"].to_numpy()
        s = raw["sales"]
        s = s[s["m"] < cur]
        piv = s.pivot_table(index="sku_id", columns="m", values="net", aggfunc="sum")
        self.M = piv.reindex(index=self.sku_ids, columns=self.months).to_numpy(dtype=float)
        gross = s.pivot_table(index="sku_id", columns="m", values="qty_sold", aggfunc="sum")
        self.G = gross.reindex(index=self.sku_ids, columns=self.months).to_numpy(dtype=float)
        # цензурирование: на 1-е число остатка нет (строки нет => 0) и продаж нет => спрос не наблюдаем.
        st = raw["stock"]
        has = pd.crosstab(st["sku_id"], st["m"]).reindex(index=self.sku_ids, columns=self.months, fill_value=0).to_numpy() > 0
        ever = np.isin(self.sku_ids, st["sku_id"].unique())          # у never_snapshotted цензурирование не определить
        late = np.array([m >= pd.Period(censor_from, "M") for m in self.months])
        cens = (~has) & (np.nan_to_num(self.M, nan=-1) == 0) & late[None, :] & ever[:, None]
        self.censored = cens
        self.valid = ~np.isnan(self.M) & (~cens if censor else True)
        self.moy = np.array([m.month - 1 for m in self.months])

    def cat_totals(self):
        df = pd.DataFrame(np.nan_to_num(self.M), index=self.sku_ids, columns=self.months)
        return df.groupby(self.cat).sum().T  # index=months, cols=categories

    def seasonal_by_row(self, upto, min_months):
        tot = self.cat_totals().iloc[:upto]
        idx = {c: seasonal_index(tot[c], min_months) for c in tot.columns}
        return idx, np.vstack([idx[c] for c in self.cat])

    def fit(self, upto, idx_rows, window=24, halflife=6.0, min_obs=3):
        """Уровень и σ по столбцам [upto-window, upto). Возвращает dict массивов (n,)."""
        lo = max(0, upto - window)
        M = self.M[:, lo:upto]
        valid = self.valid[:, lo:upto]
        seas = np.maximum(idx_rows[:, self.moy[lo:upto]], 1e-6)
        age = (upto - 1) - np.arange(lo, upto)
        w = 0.5 ** (age / halflife)
        ww = valid * w[None, :]
        # ratio-оценка: level = Σw·M / Σw·seas по валидным месяцам (устойчива при seas≈0 в межсезонье)
        sw = (ww * seas).sum(1)
        level = np.where(sw > 0, (np.nan_to_num(M) * ww).sum(1) / np.maximum(sw, 0.5), 0.0)
        nobs = valid.sum(1)
        low = nobs < min_obs
        nn = ~np.isnan(M)
        sa = (seas * nn).sum(1)
        lvl_all = np.where(nn.sum(1) > 0, np.nan_to_num(M).sum(1) / np.maximum(sa, 0.5), 0.0)
        level = np.where(low, lvl_all, level)
        # σ в сезонно-нормированных единицах: Var(спрос) ∝ среднее => r=(M-level·seas)/sqrt(seas); σ_период = σ_u·sqrt(Σseas)
        resid = np.where(valid, (np.nan_to_num(M) - level[:, None] * seas) / np.sqrt(np.maximum(seas, 0.05)), 0.0)
        sigma = np.sqrt((resid ** 2).sum(1) / np.maximum(nobs, 1))
        sigma = np.where(nobs < 6, np.maximum(sigma, np.sqrt(np.maximum(level, 0))), sigma)
        cens12 = self.censored[:, max(0, upto - 12):upto].sum(1)
        return {"level": level, "sigma": sigma, "nobs": nobs, "lower_bound": low & (cens12 > 0),
                "cens12": cens12, "hist_months": nn.sum(1)}
