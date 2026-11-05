"""Ядро: прогноз -> проекция остатка -> safety stock -> qty / даты / статусы / приоритет. Только расчёт, без LLM."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from . import data as D
from .forecast import Prepared

warnings.filterwarnings("ignore", category=RuntimeWarning)


def plan_math(s0, cumD, cumA, L, C, ss, review_days, H):
    """Формулы (docs/FORMULAS.md). cumD/cumA: накопленный спрос/приход на днях 0..H (длина H+1).
    P(t)=s0+cumA(t)-cumD(t)  — проекция остатка c учётом заказов в пути.
    reorder-триггер: первый день t_ss, когда P(t) <= SS  =>  дата заказа = t_ss - L (не 'сегодня+lead').
    target = Σспроса[L, L+C] + SS ;  qty = max(0, target - max(0, P(L)))  — для заказа, размещаемого в tau."""
    t = np.arange(H + 1)
    P = s0 + cumA - cumD
    has_demand = cumD[-1] >= 1.0
    # день исчерпания (с учётом pipeline / только остаток)
    def first_zero(arr):
        if not has_demand:
            return None
        i = np.nonzero((arr <= -1.0) & (t > 0))[0]   # дефицит = ожидаемый неудовлетворённый спрос >= 1 шт
        return int(i[0]) if len(i) else None
    dso_pipe = first_zero(P)
    dso_stock = first_zero(s0 - cumD)
    # reorder
    tau = None
    if has_demand:
        if s0 <= ss:
            t_ss = 0
        else:
            i = np.nonzero((P <= ss) & (t > 0))[0]
            t_ss = int(i[0]) if len(i) else None
        if t_ss is not None:
            tau = max(0, t_ss - L)
    qty_now = _qty(s0, cumD, cumA, P, 0, L, C, ss, H)
    qty_at_due = _qty(s0, cumD, cumA, P, tau, L, C, ss, H) if tau is not None else 0.0
    return {"days_to_stockout": dso_pipe, "days_to_stockout_stock_only": dso_stock, "tau": tau,
            "qty_now": qty_now, "qty_at_due": qty_at_due,
            "proj_at_arrival": max(0.0, float(P[min(L, H)])),
            "demand_L": float(cumD[min(L, H)]), "demand_cover": float(cumD[min(L + C, H)] - cumD[min(L, H)]),
            "cover_formula": None}


def _qty(s0, cumD, cumA, P, tau, L, C, ss, H):
    if tau is None:
        return 0.0
    a = min(tau + L, H)
    e = min(tau + L + C, H)
    target = cumD[e] - cumD[a] + ss
    return float(max(0.0, target - max(0.0, P[a])))


def build_calendar(asof, H):
    days = pd.date_range(pd.Timestamp(asof) + pd.Timedelta(days=1), periods=H)
    return days, np.array([d.year * 12 + d.month for d in days]), np.array([d.days_in_month for d in days])


def run_plan(raw, cfg, asof):
    asof = pd.Timestamp(asof)
    cur = asof.to_period("M")
    H = int(cfg["horizon_days"])
    fc = cfg["forecast"]
    z = norm.ppf(cfg["service_level"])
    P = Prepared(raw, cur, fc["censor_from"], censor=fc.get("use_censoring", False))
    idx, idx_rows = P.seasonal_by_row(len(P.months), fc["seasonal_min_months"])
    fit = P.fit(len(P.months), idx_rows, fc["window_months"], fc["halflife_months"], fc["min_obs"])
    level = fit["level"] * fc.get("demand_scale", 1.0)
    n = len(P.sku_ids)

    # --- ежедневный прогноз
    days, ym, dim = build_calendar(asof, H)
    pos = ym - (cur.year * 12 + cur.month)
    moy_days = np.array([d.month - 1 for d in days])
    daily = np.zeros((n, H + 1))
    daily[:, 1:] = level[:, None] * idx_rows[:, moy_days] / dim[None, :]
    cumD = np.cumsum(daily, axis=1)

    # --- остатки и pipeline
    stock = D.roll_forward_stock(raw, asof).set_index("sku_id").reindex(P.sku_ids)
    pos_po = D.classify_open_pos(raw["po"], asof, cfg)
    cumA = np.zeros((n, H + 1))
    row_of = {s: i for i, s in enumerate(P.sku_ids)}
    arr = np.zeros((n, H + 2))
    for r in pos_po[pos_po["factor"] > 0].itertuples():
        arr[row_of[r.sku_id], min(int(r.arrival_offset), H + 1)] += r.qty_effective
    cumA = np.cumsum(arr[:, :H + 1], axis=1)
    pip = pos_po.groupby("sku_id").agg(pipeline_nominal=("qty_pending", "sum"), pipeline_eff=("qty_effective", "sum"))
    stale = pos_po[pos_po["state"] == "stale"].groupby("sku_id")["qty_pending"].sum()
    late = pos_po[pos_po["state"] == "late"].groupby("sku_id")["qty_pending"].sum()

    lead = D.lead_times(raw, cfg, asof).set_index("sku_id").reindex(P.sku_ids)
    lots = D.lot_rules(raw, cfg, asof).set_index("sku_id").reindex(P.sku_ids)
    sku = raw["sku"].set_index("sku_id").reindex(P.sku_ids)

    # --- рентабельность: sku_dim -> пересчёт из продаж -> «нет себестоимости»
    s = raw["sales"]
    l12 = s[(s["m"] >= cur - 12) & (s["m"] < cur)]
    sold12 = l12.groupby("sku_id")["qty_sold"].sum().reindex(P.sku_ids).fillna(0)
    mk = l12[l12["markup_pct"].notna() & (l12["qty_sold"] > 0)].copy()
    mk["w"] = mk["markup_pct"] * mk["qty_sold"]
    mk_calc = (mk.groupby("sku_id")["w"].sum() / mk.groupby("sku_id")["qty_sold"].sum()).reindex(P.sku_ids)
    markup = sku["avg_markup_pct_12m"].copy()
    markup = markup.fillna(mk_calc)
    dm = int(cfg["dead_months"])
    gross_dm = np.nan_to_num(P.G[:, -dm:]).sum(1)

    rows = []
    for i, sid in enumerate(P.sku_ids):
        cat = P.cat[i]
        L = int(lead.iloc[i]["lead_days"])
        C = int(cfg["cover_days_by_category"].get(cat, cfg["cover_default_days"]))
        s0 = float(stock.iloc[i]["stock_asof"])
        sigma = fit["sigma"][i]
        Lw = min(L + cfg["lead"]["slip_buffer_days"], H)
        ss = z * sigma * np.sqrt(float((idx_rows[i, moy_days[:Lw]] / dim[:Lw]).sum()))   # Σ сезонно-взвешенных месяцев за L+slip
        if level[i] <= 0:
            ss = 0.0
        m = plan_math(s0, cumD[i], cumA[i], L, C, ss, cfg["review_days"], H)
        total_pipe = s0 + float(pip["pipeline_eff"].get(sid, 0.0))
        avg_daily = (cumD[i][min(L + C, H)]) / max(min(L + C, H), 1)
        cover_formula = total_pipe / avg_daily if avg_daily > 0 else None
        dso = m["days_to_stockout"]
        cover_pipe = dso if dso is not None else (H if avg_daily > 0 else None)
        cover_stock = m["days_to_stockout_stock_only"] if m["days_to_stockout_stock_only"] is not None else (H if avg_daily > 0 else None)
        lot = int(lots.iloc[i]["lot"]); mb = float(lots.iloc[i]["min_batch"])
        tau = m["tau"]
        qty_basis = m["qty_now"] if (tau is not None and tau <= 0) else m["qty_at_due"]
        qty_final = D.apply_lot(qty_basis, mb, lot)
        mkv = markup.iloc[i]
        w = sku.iloc[i]["weight_kg"]
        flags = []
        sold_recent = sold12.iloc[i] > 0
        st_state = stock.iloc[i]["stock_state"]
        if st_state == "never_snapshotted":
            flags.append("STOCK_UNVERIFIED")
        elif st_state == "row_absent_zero":
            flags.append("STOCK_ROW_ABSENT_ZERO")
        if stale.get(sid, 0) > 0: flags.append("STALE_PO")
        if late.get(sid, 0) > 0: flags.append("LATE_PO")
        if pd.isna(w): flags.append("NO_WEIGHT")
        if not bool(lots.iloc[i]["moq_known"]): flags.append("MOQ_UNKNOWN")
        if fit["hist_months"][i] < 12: flags.append("SHORT_HISTORY")
        if fit["lower_bound"][i]: flags.append("DEMAND_LOWER_BOUND")
        if lead.iloc[i]["lead_src"] == "default": flags.append("LEAD_DEFAULT")
        if mb > qty_basis > 0: flags.append("MOQ_UPLIFT")   # поднято до типичной партии

        # --- статус риска
        if level[i] <= 0 or not sold_recent and gross_dm[i] == 0:
            risk = "N/A"
        elif dso is not None and dso < L:
            risk = "SHORTAGE"
        elif cover_pipe is not None and cover_pipe > L + C + cfg["excess_buffer_days"] and (dso is None or True) \
                and cover_formula is not None and cover_formula > L + C + cfg["excess_buffer_days"]:
            risk = "EXCESS"
        else:
            risk = "OK"

        # --- действие (приоритет правил сверху вниз)
        reasons = []
        if gross_dm[i] == 0 and (s0 > 0 or total_pipe > s0):
            action, reasons = "DEAD_STOCK", [f"NO_SALES_{dm}M", "STOCK_GT_0"]
        elif gross_dm[i] == 0:
            action, reasons = "INACTIVE", [f"NO_SALES_{dm}M", "NO_STOCK"]
        elif pd.isna(mkv):
            action, reasons = "MANUAL_CHECK", ["COST_MISSING"]
        elif mkv < cfg["min_markup_pct"]:
            action, reasons = "REVIEW_ASSORTMENT", ["LOW_MARKUP"]
        elif st_state == "never_snapshotted":
            action, reasons = "MANUAL_CHECK", ["STOCK_UNVERIFIED"]
        elif qty_final > 0 and tau is not None and tau <= cfg["review_days"]:
            action = "BUY_NOW"; reasons = ["REORDER_POINT_WITHIN_REVIEW" if tau > 0 else "REORDER_POINT_REACHED"]
        elif qty_final > 0:
            action, reasons = "BUY_LATER", ["ORDER_DATE_AFTER_REVIEW"]
        else:
            action, reasons = "NO_ACTION", ["COVERED"]
        if action in ("MANUAL_CHECK", "REVIEW_ASSORTMENT") and qty_final > 0 and tau is not None and tau <= cfg["review_days"]:
            reasons.append("WOULD_BUY")
        if risk == "SHORTAGE": reasons.append("STOCKOUT_BEFORE_LEAD")
        if risk == "EXCESS": reasons.append("COVER_ABOVE_L_PLUS_COVER_PLUS_X")
        if "STALE_PO" in flags: reasons.append("STALE_PO_IGNORED")

        # --- приоритет: urgency × log(объём) × вес цены
        urg = 1.0 if dso is not None and dso <= 0 else (max(0.0, min(1.0, 1 - dso / (L + 30))) if dso is not None else 0.0)
        pdc = sku.iloc[i]["price_decile"]
        pw = cfg["priority"]["price_weight_base"] + cfg["priority"]["price_weight_step"] * (pdc if pd.notna(pdc) else 5)
        uar = max(0.0, m["demand_L"] - s0 - float(cumA[i][min(L, H)]))      # недопокрытый спрос за плечо, шт
        exposure = uar + max(0.0, m["demand_cover"] - m["proj_at_arrival"])
        prio = urg * pw * np.log1p(exposure) if qty_final > 0 else 0.0
        if mb > 0 and qty_basis > 0 and qty_basis < 0.25 * mb: reasons.append("MOQ_DOMINATES")
        due_date = (asof + pd.Timedelta(days=int(max(tau, 0)))).date().isoformat() if tau is not None else None
        rows.append(dict(
            sku_id=sid, category=cat, supplier_id=sku.iloc[i]["supplier_id"], supply_type=sku.iloc[i]["supply_type"],
            price_decile=pdc, action=action, risk=risk, reason_codes="|".join(reasons), flags="|".join(flags),
            qty_recommended=qty_final if action in ("BUY_NOW", "BUY_LATER") else 0.0,
            qty_would_buy=qty_final, qty_raw=round(qty_basis, 1), next_order_in_days=tau, order_date=due_date,
            priority_raw=prio, stock_asof=s0, stock_state=st_state, snapshot=float(stock.iloc[i]["snapshot"]),
            sold_partial=float(stock.iloc[i]["sold_partial"]), rc_partial=float(stock.iloc[i]["rc_partial"]),
            pipeline_nominal=float(pip["pipeline_nominal"].get(sid, 0.0)), pipeline_eff=round(float(pip["pipeline_eff"].get(sid, 0.0)), 1),
            stale_po_qty=float(stale.get(sid, 0.0)),
            lead_days=L, lead_src=lead.iloc[i]["lead_src"], cover_policy_days=C,
            cover_stock_only_days=cover_stock, cover_with_pipeline_days=cover_pipe,
            cover_formula_days=None if cover_formula is None else round(cover_formula, 1),
            days_to_stockout=dso, forecast_L=round(m["demand_L"], 1), forecast_cover=round(m["demand_cover"], 1),
            units_at_risk=round(uar, 1), safety_stock=round(ss, 1), target=round(m["demand_cover"] + ss, 1), proj_at_arrival=round(m["proj_at_arrival"], 1),
            level_m=round(float(level[i]), 2), sigma_m=round(float(sigma), 2), z=round(float(z), 3),
            fc_next_3m=round(float(sum(level[i] * idx_rows[i, (cur.month - 1 + k) % 12] for k in range(3))), 1),
            markup_pct=None if pd.isna(mkv) else float(mkv), lot=lot, min_batch=mb,
            weight_kg=None if pd.isna(w) else float(w), kg_order=None if pd.isna(w) else float(w) * qty_final,
            sold_12m=float(sold12.iloc[i]), gross_dead_window=float(gross_dm[i]),
        ))
    rec = pd.DataFrame(rows)
    mx = rec["priority_raw"].max() or 1.0
    rec["priority"] = (100 * rec["priority_raw"] / mx).round(1)
    rec = rec.drop(columns="priority_raw")
    meta = {"as_of": str(asof.date()), "season_index": {k: [round(float(x), 3) for x in v] for k, v in idx.items()},
            "z": float(z), "open_po_rows": int(len(pos_po)),
            "open_po_states": pos_po["state"].value_counts().to_dict() if len(pos_po) else {},
            "open_po_pending_total": float(pos_po["qty_pending"].sum()), "open_po_effective_total": float(pos_po["qty_effective"].sum()),
            "on_order_qty_check": float(raw["sku"]["on_order_qty"].sum())}
    return rec, pos_po, meta
