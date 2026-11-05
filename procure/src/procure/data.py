"""Загрузка, единая дата среза, roll-forward остатков, открытые PO, лид-тайм, MOQ-прокси."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

FILES = ["sku_dim", "sales_monthly", "stock_monthly", "purchase_orders", "receipts", "supplier_dim"]


def load_raw(data_dir):
    d = Path(data_dir)
    raw = {
        "sku": pd.read_csv(d / "sku_dim.csv"),
        "sales": pd.read_csv(d / "sales_monthly.csv"),
        "stock": pd.read_csv(d / "stock_monthly.csv"),
        "po": pd.read_csv(d / "purchase_orders.csv", parse_dates=["order_date", "promised_date"]),
        "rc": pd.read_csv(d / "receipts.csv", parse_dates=["receipt_date"]),
        "sup": pd.read_csv(d / "supplier_dim.csv"),
    }
    s = raw["sales"]
    s["m"] = pd.PeriodIndex(s["month"], freq="M")
    # ВОЗВРАТЫ: прогнозируем net = sold - returned (возврат возвращается на склад;
    # roll-forward на данных сходится точнее с net: 63.6% vs 58.5% точных совпадений).
    s["net"] = (s["qty_sold"] - s["qty_returned"]).clip(lower=0)
    raw["stock"]["m"] = pd.PeriodIndex(raw["stock"]["month"], freq="M")
    return raw


def file_hashes(data_dir):
    out = {}
    for f in FILES:
        p = Path(data_dir) / f"{f}.csv"
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


# ---------------------------------------------------------------- остатки
def roll_forward_stock(raw, asof):
    """stock(as_of) = snapshot(1-е число месяца as_of) - net_sales(1..as_of) + receipts(1..as_of).

    ПРАВИЛО: экспорт stock_monthly «zero-suppressed» — строк со stock=0 нет вообще
    (проверено: stock_qty==0 встречается 0 раз; при отсутствии строки roll-forward даёт <=0 в 80% случаев).
    Поэтому отсутствие строки = 0, но с флагом статуса:
      snapshot          — строка на 1-е число есть
      row_absent_zero   — у SKU есть снимки в другие месяцы, на этот нет -> 0 (подтверждено динамикой)
      never_snapshotted — снимков нет вообще -> 0, но НЕ ВЕРИФИЦИРОВАНО (ручная проверка, если есть спрос)
    """
    asof = pd.Timestamp(asof)
    cur = asof.to_period("M")
    if cur < pd.Period("2024-01", "M"):
        raise ValueError("as_of раньше первого снимка остатка (2024-01)")
    st = raw["stock"]
    snap = st[st["m"] == cur].set_index("sku_id")["stock_qty"]
    ever = set(st["sku_id"])
    out = raw["sku"][["sku_id"]].copy()
    out["snapshot"] = out["sku_id"].map(snap)
    has = out["snapshot"].notna()
    out["stock_state"] = np.where(has, "snapshot",
                                  np.where(out["sku_id"].isin(ever), "row_absent_zero", "never_snapshotted"))
    out["snapshot"] = out["snapshot"].fillna(0.0)
    if asof.day > 1:  # продажи неполного месяца: только для roll-forward, в фит не идут
        sp = raw["sales"][raw["sales"]["m"] == cur].groupby("sku_id")["net"].sum()
    else:
        sp = pd.Series(dtype=float)
    rc = raw["rc"]
    rp = rc[(rc["receipt_date"] >= cur.start_time) & (rc["receipt_date"] <= asof)].groupby("sku_id")["qty"].sum()
    out["sold_partial"] = out["sku_id"].map(sp).fillna(0.0)
    out["rc_partial"] = out["sku_id"].map(rp).fillna(0.0)
    out["stock_asof"] = (out["snapshot"] - out["sold_partial"] + out["rc_partial"]).clip(lower=0)
    return out


# ---------------------------------------------------------------- открытые заказы
def supplier_reliability(po, asof, cfg):
    """Доля исполнения по терминальным PO (completed/closed), со сжатием к общему среднему."""
    c = cfg["po"]
    t = po[(po["status"] != "open") & (po["order_date"] <= asof) & (po["qty_ordered"] > 0)].copy()
    t["fill"] = (t["qty_received"] / t["qty_ordered"]).clip(upper=1)
    g = t["fill"].mean() if len(t) else 1.0
    agg = t.groupby("supplier_id")["fill"].agg(["sum", "count"])
    rel = (agg["sum"] + c["reliability_k"] * g) / (agg["count"] + c["reliability_k"])
    return rel.clip(lower=c["reliability_floor"], upper=1.0), max(min(g, 1.0), c["reliability_floor"])


def classify_open_pos(po, asof, cfg):
    """Политика по открытым PO (qty_pending берётся ТОЛЬКО отсюда, не из sku_dim.on_order_qty — иначе дубль).
    on_time : promised >= as_of           -> приход в promised, вес = надёжность поставщика
                                             (1.0, если поставщик подтвердил qty_confirmed >= pending)
    late    : просрочка <= stale_days     -> приход as_of+late_assumed_arrival, вес = надёжность * late_factor
    stale   : просрочка >  stale_days     -> НЕ учитываем, флаг STALE_PO (закупщик закрывает/подтверждает)
    """
    c = cfg["po"]
    asof = pd.Timestamp(asof)
    rel, g = supplier_reliability(po, asof, cfg)
    o = po[(po["status"] == "open") & (po["qty_pending"] > 0) & (po["order_date"] <= asof)].copy()
    o["days_overdue"] = (asof - o["promised_date"]).dt.days
    o["state"] = np.where(o["days_overdue"] <= 0, "on_time",
                          np.where(o["days_overdue"] <= c["stale_days"], "late", "stale"))
    o["reliability"] = o["supplier_id"].map(rel).fillna(g)
    confirmed = o["qty_confirmed"] >= o["qty_pending"]
    base = np.where(confirmed, 1.0, o["reliability"])
    o["factor"] = np.where(o["state"] == "on_time", base,
                           np.where(o["state"] == "late", base * c["late_factor"], 0.0))
    o["arrival_date"] = np.where(o["state"] == "on_time", o["promised_date"],
                                 asof + pd.Timedelta(days=c["late_assumed_arrival_days"]))
    o["arrival_date"] = pd.to_datetime(o["arrival_date"])
    o["arrival_offset"] = np.maximum(1, (o["arrival_date"] - asof).dt.days)
    o["qty_effective"] = o["qty_pending"] * o["factor"]
    return o


# ---------------------------------------------------------------- лид-тайм
def lead_times(raw, cfg, asof):
    c = cfg["lead"]
    asof = pd.Timestamp(asof)
    po = raw["po"]
    po = po[(po["order_date"] <= asof) & (po["planned_lead_days"] > 0)]
    po = po[po["order_date"] >= asof - pd.DateOffset(months=c["history_months"])]
    sku = raw["sku"][["sku_id", "supplier_id", "supply_type"]].copy()
    by_sku = po.groupby("sku_id")["planned_lead_days"].agg(["median", "count"])
    by_sup = po.groupby("supplier_id")["planned_lead_days"].agg(["median", "count"])
    sd = raw["sup"].set_index("supplier_id")["median_planned_lead_days"]
    L, src = [], []
    for r in sku.itertuples():
        if r.sku_id in by_sku.index and by_sku.loc[r.sku_id, "count"] >= 2:
            L.append(by_sku.loc[r.sku_id, "median"]); src.append("sku_po")
        elif r.supplier_id in by_sup.index and by_sup.loc[r.supplier_id, "count"] >= 3:
            L.append(by_sup.loc[r.supplier_id, "median"]); src.append("supplier_po")
        elif r.supplier_id in sd.index and pd.notna(sd[r.supplier_id]):
            L.append(sd[r.supplier_id]); src.append("supplier_dim")
        else:
            L.append(c["default_local_days"] if r.supply_type == "local" else c["default_import_days"])
            src.append("default")
    sku["lead_days"] = np.clip(np.round(L), c["min_days"], c["cap_days"]).astype(int)
    sku["lead_src"] = src
    return sku[["sku_id", "lead_days", "lead_src"]]


# ---------------------------------------------------------------- MOQ / партионность
def lot_rules(raw, cfg, asof):
    """Прокси MOQ из истории PO: min_batch = p25 прошлых qty_ordered; lot = наибольшее из lots, кратное >= lot_share заказов."""
    c = cfg["moq"]
    po = raw["po"][raw["po"]["order_date"] <= pd.Timestamp(asof)]
    rows = []
    for sid, g in po.groupby("sku_id"):
        q = g["qty_ordered"].to_numpy()
        if len(q) < c["min_po_history"]:
            continue
        lot = 1
        for L in c["lots"]:
            if (q % L == 0).mean() >= c["lot_share"]:
                lot = L
                break
        rows.append((sid, float(np.percentile(q, 25)), lot, len(q)))
    r = pd.DataFrame(rows, columns=["sku_id", "min_batch", "lot", "po_n"])
    out = raw["sku"][["sku_id"]].merge(r, on="sku_id", how="left")
    out["moq_known"] = out["min_batch"].notna()
    out["min_batch"] = out["min_batch"].fillna(0.0)
    out["lot"] = out["lot"].fillna(1).astype(int)
    return out


def apply_lot(qty, min_batch, lot):
    if qty <= 0:
        return 0.0
    q = max(qty, min_batch)
    return float(np.ceil(q / lot) * lot)
