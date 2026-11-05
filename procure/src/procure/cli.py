"""CLI: procure run | backtest | sensitivity | explain SKU | selftest.  Расчёт детерминирован, LLM не участвует."""
import argparse, json, sys, time
from pathlib import Path
import pandas as pd
from . import __version__, backtest as BT, data as D, engine, report
from .config import cfg_hash, load_config


def _prev_run(runs_dir, current_id):
    p = Path(runs_dir)
    if not p.exists():
        return None, None
    dirs = sorted([d for d in p.iterdir() if d.is_dir() and d.name != current_id and (d / "recommendations.csv").exists()])
    if not dirs:
        return None, None
    return dirs[-1].name, pd.read_csv(dirs[-1] / "recommendations.csv")


def cmd_run(a):
    cfg = load_config(a.config)
    asof = a.as_of or cfg["as_of"]
    raw = D.load_raw(cfg["paths"]["data_dir"])
    hashes = D.file_hashes(cfg["paths"]["data_dir"])
    runs = Path(cfg["paths"]["runs_dir"])
    # дисциплина: ежедневный запуск на тех же месячных данных не даёт нового прогноза
    last = sorted([d for d in runs.glob("*") if (d / "run_manifest.json").exists()]) if runs.exists() else []
    if last and not a.force:
        m = json.loads((last[-1] / "run_manifest.json").read_text())
        if m.get("input_hashes") == hashes and m.get("as_of") == str(asof) and m.get("config_hash") == cfg_hash(cfg):
            print(f"Входные данные, as-of и конфиг не изменились с {m['run_id']} — пересчёт пропущен (--force чтобы выполнить).")
            return 0
    run_id = f"{pd.Timestamp(asof):%Y%m%d}_{time.strftime('%H%M%S')}"
    t0 = time.time()
    rec, pos, meta = engine.run_plan(raw, cfg, asof)
    out = runs / run_id
    out.mkdir(parents=True, exist_ok=True)
    prev_id, prev = _prev_run(runs, run_id)
    rec.to_csv(out / "recommendations.csv", index=False)
    sup, lines = report.supplier_packages(rec, cfg)
    sup.to_csv(out / "supplier_packages.csv", index=False); lines.to_csv(out / "supplier_order_lines.csv", index=False)
    cal = report.order_calendar(rec); cal.to_csv(out / "order_calendar.csv", index=False)
    dl = report.delta_vs_previous(rec, prev); dl.to_csv(out / "delta.csv", index=False)
    pos.to_csv(out / "open_po_policy.csv", index=False)
    for name, acts in (("assortment_review", ["REVIEW_ASSORTMENT"]), ("manual_check", ["MANUAL_CHECK"]), ("dead_stock", ["DEAD_STOCK"])):
        rec[rec["action"].isin(acts)].to_csv(out / f"{name}.csv", index=False)
    bt, sens = pd.DataFrame(), pd.DataFrame()
    bt_path = runs / "backtest_latest.csv"
    if bt_path.exists():
        bt = pd.read_csv(bt_path)
    sp = runs / "sensitivity_latest.csv"
    if sp.exists():
        sens = pd.read_csv(sp)
    flags = pd.Series([f for x in rec["flags"].fillna("") for f in x.split("|") if f]).value_counts().rename_axis("flag").reset_index(name="n")
    manifest = {"run_id": run_id, "as_of": str(asof), "code_version": __version__, "config_hash": cfg_hash(cfg), "input_hashes": hashes,
                "rows": {k: int(len(v)) for k, v in raw.items()}, "counts": rec["action"].value_counts().to_dict(),
                "risk_counts": rec["risk"].value_counts().to_dict(), "prev_run": prev_id, "runtime_sec": round(time.time() - t0, 1),
                "freshness": {"last_full_sales_month": str(pd.Timestamp(asof).to_period("M") - 1), "stock_snapshot_month": str(pd.Timestamp(asof).to_period("M"))}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    keep = rec[rec["action"] != "INACTIVE"]
    payload = {"meta": meta, "manifest": manifest, "cfg": cfg, "rec": keep.where(keep.notna(), None).to_dict("records"),
               "suppliers": sup.to_dict("records"), "calendar": cal.to_dict("records"), "delta": dl.to_dict("records"),
               "backtest": bt[bt["H"] == 6].to_dict("records") if len(bt) else [], "sens": sens.to_dict("records"), "flagCounts": flags.to_dict("records")}
    report.write_dashboard(out / "dashboard.html", payload)
    (runs / "LATEST").write_text(run_id)
    print(f"run {run_id}: {manifest['counts']}\n-> {out}/dashboard.html")
    return 0


def cmd_backtest(a):
    cfg = load_config(a.config); raw = D.load_raw(cfg["paths"]["data_dir"])
    cur = pd.Timestamp(a.as_of or cfg["as_of"]).to_period("M")
    df = BT.run_backtest(raw, cfg, cur, censor=cfg["forecast"].get("use_censoring", False))
    ov, bc = BT.summarize(df)
    runs = Path(cfg["paths"]["runs_dir"]); runs.mkdir(exist_ok=True)
    ov.to_csv(runs / "backtest_latest.csv", index=False); bc.to_csv(runs / "backtest_by_category.csv", index=False)
    print(ov[ov["H"] == 6].round(3).to_string(index=False))
    return 0


def cmd_sensitivity(a):
    cfg = load_config(a.config); raw = D.load_raw(cfg["paths"]["data_dir"])
    s = report.sensitivity(raw, cfg, a.as_of or cfg["as_of"], engine.run_plan)
    runs = Path(cfg["paths"]["runs_dir"]); runs.mkdir(exist_ok=True)
    s.to_csv(runs / "sensitivity_latest.csv", index=False); print(s.round(0).to_string(index=False))
    return 0


def cmd_explain(a):
    cfg = load_config(a.config); runs = Path(cfg["paths"]["runs_dir"])
    rid = (runs / "LATEST").read_text().strip()
    r = pd.read_csv(runs / rid / "recommendations.csv")
    r = r[r["sku_id"] == a.sku]
    if r.empty:
        print("SKU не найден"); return 1
    print(r.iloc[0].to_json(indent=2, force_ascii=False)); return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="procure")
    p.add_argument("--config", default="config.yaml")
    sp = p.add_subparsers(dest="cmd", required=True)
    for n, fn in (("run", cmd_run), ("backtest", cmd_backtest), ("sensitivity", cmd_sensitivity)):
        s = sp.add_parser(n); s.add_argument("--as-of"); s.set_defaults(fn=fn)
        if n == "run": s.add_argument("--force", action="store_true")
    e = sp.add_parser("explain"); e.add_argument("sku"); e.set_defaults(fn=cmd_explain)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
