"""Агрегаты для закупщиков: пакеты по поставщикам, календарь, контейнеры, дельта к прошлому запуску, чувствительность, дашборд."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import deep_update


def supplier_packages(rec, cfg):
    buy = rec[rec["action"] == "BUY_NOW"].copy()
    buy["supplier_id"] = buy["supplier_id"].fillna("UNSPECIFIED")
    pay = cfg["container"]["payload_kg"]
    g = buy.groupby("supplier_id").agg(skus=("sku_id", "count"), units=("qty_recommended", "sum"),
                                      kg=("kg_order", "sum"), skus_no_weight=("weight_kg", lambda x: int(x.isna().sum())),
                                      shortage_skus=("risk", lambda x: int((x == "SHORTAGE").sum())),
                                      max_priority=("priority", "max")).reset_index()
    g["containers_by_weight_lb"] = (g["kg"] / pay).round(2)
    return g.sort_values("max_priority", ascending=False), buy.sort_values(["supplier_id", "priority"], ascending=[True, False])


def buy_now_by_supplier(rec):
    """Плоская выгрузка BUY_NOW: поставщик -> категория -> SKU -> количество."""
    r = rec[rec["action"] == "BUY_NOW"].copy()
    r["supplier_id"] = r["supplier_id"].fillna("UNSPECIFIED")
    r["qty_recommended"] = r["qty_recommended"].round(0).astype(int)
    return r.sort_values(["supplier_id", "category", "sku_id"])[["supplier_id", "category", "sku_id", "qty_recommended"]]


def order_calendar(rec):
    r = rec[rec["action"].isin(["BUY_NOW", "BUY_LATER"]) & rec["order_date"].notna()].copy()
    d = pd.to_datetime(r["order_date"])
    r["week"] = d.dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    return r.groupby("week").agg(skus=("sku_id", "count"), units=("qty_would_buy", "sum"), kg=("kg_order", "sum")).reset_index()


def delta_vs_previous(rec, prev):
    if prev is None:
        return pd.DataFrame(columns=["sku_id", "change", "detail"])
    m = rec.merge(prev[["sku_id", "action", "qty_recommended", "risk"]], on="sku_id", how="left", suffixes=("", "_prev"))
    out = []
    for r in m.itertuples():
        if r.action != r.action_prev:
            out.append((r.sku_id, "ACTION_CHANGED", f"{r.action_prev} -> {r.action}"))
        elif r.action in ("BUY_NOW", "BUY_LATER") and abs(r.qty_recommended - (r.qty_recommended_prev or 0)) > 0.1 * max(r.qty_recommended_prev or 0, 1):
            out.append((r.sku_id, "QTY_CHANGED", f"{r.qty_recommended_prev:.0f} -> {r.qty_recommended:.0f}"))
        elif r.risk != r.risk_prev and r.risk in ("SHORTAGE", "EXCESS"):
            out.append((r.sku_id, "RISK_CHANGED", f"{r.risk_prev} -> {r.risk}"))
    return pd.DataFrame(out, columns=["sku_id", "change", "detail"])


def sensitivity(raw, cfg, asof, run_plan):
    variants = [("base", {})]
    for k in (30, 60, 90, 120):
        variants.append((f"cover_all={k}d", {"cover_days_by_category": {c: k for c in cfg["cover_days_by_category"]}, "cover_default_days": k}))
    for sl in (0.90, 0.98):
        variants.append((f"service_level={sl}", {"service_level": sl}))
    variants += [("demand_scale=0.85", {"forecast.demand_scale": 0.85}), ("trust_all_open_PO", {"po.reliability_floor": 1.0, "po.late_factor": 1.0, "po.stale_days": 9999}),
                 ("ignore_open_PO", {"po.reliability_floor": 0.0, "po.late_factor": 0.0, "po.stale_days": -1})]
    rows = []
    for name, ov in variants:
        c = deep_update(cfg, **{k: v for k, v in ov.items()})
        r, _, _ = run_plan(raw, c, asof)
        b = r[r["action"] == "BUY_NOW"]
        rows.append((name, len(b), float(b["qty_recommended"].sum()), float(b["kg_order"].sum()),
                     int((r["risk"] == "SHORTAGE").sum()), int((r["risk"] == "EXCESS").sum())))
    return pd.DataFrame(rows, columns=["variant", "buy_now_skus", "buy_now_units", "buy_now_kg", "shortage_skus", "excess_skus"])


def write_dashboard(path, payload):
    tpl = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    Path(path).write_text(tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=_js)), encoding="utf-8")


def _js(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    return str(o)
