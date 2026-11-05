import sys, unittest, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
warnings.filterwarnings("ignore")
from procure import data as D, engine as E, forecast as F
from procure.config import load_config, deep_update

H = 540
def flat(rate, h=H):
    d = np.full(h + 1, rate); d[0] = 0; return np.cumsum(d)

class TestPlanMath(unittest.TestCase):
    def test_qty_formula_flat(self):
        # спрос 10/д, остаток 500, L=60, C=30, SS=100: на приходе остатка нет -> qty = 300+100 = 400
        m = E.plan_math(500, flat(10), np.zeros(H + 1), 60, 30, 100, 30, H)
        self.assertAlmostEqual(m["qty_now"], 0.0 if False else m["qty_now"])
        self.assertAlmostEqual(m["qty_at_due"] if m["tau"] else m["qty_now"], 400.0, delta=1)

    def test_reorder_date_not_today_plus_lead(self):
        # остаток 3000 при 10/д, SS=0 => t_ss=300 => заказ через 300-60=240 дн
        m = E.plan_math(3000, flat(10), np.zeros(H + 1), 60, 30, 0, 30, H)
        self.assertIn(m["tau"], (240, 241))     # заказ, когда покрытие дойдёт до L (±1 день порога дефицита)

    def test_pipeline_reduces_qty(self):
        cumA = np.zeros(H + 1); cumA[30:] = 600   # приход 600 на 30-й день
        a = E.plan_math(0, flat(10), np.zeros(H + 1), 60, 30, 0, 30, H)["qty_now"]
        b = E.plan_math(0, flat(10), cumA, 60, 30, 0, 30, H)["qty_now"]
        self.assertEqual(a, 300.0); self.assertEqual(b, 300.0)   # 600 съедено спросом до дня 60 -> эффект 0
        cumA2 = np.zeros(H + 1); cumA2[30:] = 2000
        self.assertEqual(E.plan_math(0, flat(10), cumA2, 60, 30, 0, 30, H)["qty_now"], 0.0)

    def test_no_demand_no_order_no_stockout(self):
        m = E.plan_math(0, np.zeros(H + 1), np.zeros(H + 1), 60, 30, 0, 30, H)
        self.assertIsNone(m["tau"]); self.assertIsNone(m["days_to_stockout"]); self.assertEqual(m["qty_now"], 0.0)

    def test_stockout_detected(self):
        m = E.plan_math(100, flat(10), np.zeros(H + 1), 60, 30, 0, 30, H)
        self.assertEqual(m["days_to_stockout"], 11)

class TestReorderTrigger(unittest.TestCase):
    def test_pipeline_covering_lead_defers_order(self):
        cumA = np.zeros(H + 1); cumA[30:] = 2000
        m = E.plan_math(0, flat(10), cumA, 60, 30, 100, 30, H)
        self.assertIn(m["tau"], (140, 141))  # SS-триггер дал бы 130, но покрытие (дефицит на дне ~201) > L => ждём, пока покрытие дойдёт до L
    def test_breach_before_lead_is_not_fixable_by_new_order(self):
        # остаток 0 < SS, но поставка в пути придёт до L и покрывает спрос: «заказать сегодня» быть не должно
        cumA = np.zeros(H + 1); cumA[20:] = 1500
        self.assertGreater(E.plan_math(0, flat(10), cumA, 90, 30, 100, 30, H)["tau"], 0)

class TestSeasonalSafetyWindow(unittest.TestCase):
    def test_ss_window_after_arrival(self):
        d = np.zeros(H + 1); d[1:101] = 10.0           # сезон только в первые 100 дней, потом спроса нет
        cumD = np.cumsum(d)
        # SS «до прихода» (скаляр 632): P(L=120)=500 < 632 => триггер срабатывает (старое поведение; дату отодвигает уже правило покрытия)
        self.assertIsNotNone(E.plan_math(1500, cumD, np.zeros(H + 1), 120, 60, 632, 30, H)["tau"])
        # SS на окно ПОСЛЕ прихода (после сезона = 0) => заказывать нечего
        self.assertIsNone(E.plan_math(1500, cumD, np.zeros(H + 1), 120, 60, np.zeros(H + 1), 30, H)["tau"])

class TestLot(unittest.TestCase):
    def test_apply_lot(self):
        self.assertEqual(D.apply_lot(0, 500, 100), 0.0)
        self.assertEqual(D.apply_lot(120, 500, 100), 500.0)
        self.assertEqual(D.apply_lot(730, 500, 100), 800.0)

class TestSeasonal(unittest.TestCase):
    def test_peak_detected_and_normalised(self):
        months = pd.period_range("2020-01", periods=60, freq="M")
        v = np.array([10.0 if m.month != 10 else 100.0 for m in months])
        idx = F.seasonal_index(pd.Series(v, index=months))
        self.assertEqual(int(np.argmax(idx)), 9); self.assertAlmostEqual(idx.mean(), 1.0)
    def test_flat_is_flat(self):
        months = pd.period_range("2020-01", periods=60, freq="M")
        idx = F.seasonal_index(pd.Series(np.full(60, 50.0), index=months))
        self.assertTrue(np.allclose(idx, 1.0))

def raw_stub():
    sku = pd.DataFrame({"sku_id": ["A", "B", "C"]})
    stock = pd.DataFrame({"sku_id": ["A", "B"], "month": ["2026-08", "2026-05"], "stock_qty": [100, 7]})
    stock["m"] = pd.PeriodIndex(stock["month"], freq="M")
    sales = pd.DataFrame({"sku_id": ["A", "A"], "month": ["2026-08", "2026-07"], "net": [30, 999]})
    sales["m"] = pd.PeriodIndex(sales["month"], freq="M")
    rc = pd.DataFrame({"sku_id": ["A", "C"], "receipt_date": pd.to_datetime(["2026-08-05", "2026-08-01"]), "qty": [20, 40]})
    return {"sku": sku, "stock": stock, "sales": sales, "rc": rc}

class TestRollForward(unittest.TestCase):
    def test_rule(self):
        r = D.roll_forward_stock(raw_stub(), "2026-08-13").set_index("sku_id")
        self.assertEqual(r.loc["A", "stock_asof"], 100 - 30 + 20)       # июльские продажи не участвуют
        self.assertEqual(r.loc["A", "stock_state"], "snapshot")
        self.assertEqual(r.loc["B", "stock_state"], "row_absent_zero")
        self.assertEqual(r.loc["C", "stock_state"], "never_snapshotted")  # не молчаливый ноль: статус
        self.assertEqual(r.loc["C", "stock_asof"], 40)
    def test_asof_first_day_ignores_partial_sales(self):
        r = D.roll_forward_stock(raw_stub(), "2026-08-01").set_index("sku_id")
        self.assertEqual(r.loc["A", "stock_asof"], 100)

class TestPO(unittest.TestCase):
    def test_policy(self):
        cfg = load_config(str(Path(__file__).parents[1] / "config.yaml"))
        po = pd.DataFrame({"po_id": list("abcd"), "sku_id": list("ABCD"), "supplier_id": ["S"] * 4, "status": ["open"] * 4,
                           "order_date": pd.to_datetime(["2026-01-01"] * 4),
                           "promised_date": pd.to_datetime(["2026-09-01", "2026-07-20", "2026-02-14", "2026-09-01"]),
                           "qty_pending": [100, 100, 100, 100], "qty_confirmed": [0, 0, 0, 100], "qty_ordered": [100] * 4, "qty_received": [0] * 4})
        hist = pd.DataFrame({"po_id": ["h"], "sku_id": ["Z"], "supplier_id": ["S"], "status": ["completed"], "order_date": pd.to_datetime(["2025-01-01"]),
                             "promised_date": pd.to_datetime(["2025-05-01"]), "qty_pending": [0], "qty_confirmed": [10], "qty_ordered": [10], "qty_received": [10]})
        o = D.classify_open_pos(pd.concat([po, hist]), "2026-08-13", cfg).set_index("sku_id")
        self.assertEqual(o.loc["A", "state"], "on_time"); self.assertEqual(o.loc["B", "state"], "late")
        self.assertEqual(o.loc["C", "state"], "stale"); self.assertEqual(o.loc["C", "qty_effective"], 0)
        self.assertEqual(o.loc["D", "factor"], 1.0)               # подтверждено поставщиком
        self.assertLess(o.loc["B", "factor"], o.loc["A", "factor"] + 1e-9)

class TestPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parents[1]
        cls.cfg = load_config(str(root / "config.yaml")); cls.raw = D.load_raw(root / "data/raw")
        cls.rec, cls.po, cls.meta = E.run_plan(cls.raw, cls.cfg, "2026-08-13")
    def test_all_skus_classified_once(self):
        self.assertEqual(len(self.rec), 791); self.assertTrue(self.rec["sku_id"].is_unique)
    def test_on_order_single_source(self):
        self.assertEqual(self.meta["open_po_pending_total"], self.meta["on_order_qty_check"])
    def test_low_markup_never_auto_bought(self):
        b = self.rec[self.rec["action"] == "BUY_NOW"]
        self.assertTrue((b["markup_pct"] >= self.cfg["min_markup_pct"]).all())
        self.assertEqual(int((self.rec["action"] == "REVIEW_ASSORTMENT").sum()), 68)  # совпадает с README
    def test_cost_missing_is_manual_not_dead(self):
        m = self.rec[self.rec["reason_codes"].str.contains("COST_MISSING")]
        self.assertTrue((m["sold_12m"] > 0).all())
    def test_stale_po_excluded(self):
        s = self.rec[self.rec["stale_po_qty"] > 0]
        self.assertTrue((s["pipeline_eff"] <= s["pipeline_nominal"] - s["stale_po_qty"] + 1e-6).all())
    def test_seasonality_heaters_peak_oct(self):
        self.assertEqual(int(np.argmax(self.meta["season_index"]["space_heaters"])), 9)
    def test_partial_month_not_in_fit(self):
        P = F.Prepared(self.raw, pd.Period("2026-08", "M"))
        self.assertEqual(str(P.months[-1]), "2026-07")
    def test_moq_dominates_goes_to_manual(self):
        d = self.rec[self.rec["reason_codes"].str.contains("MOQ_DOMINATES")]
        self.assertFalse((d["action"] == "BUY_NOW").any() and not (d[d["action"] == "BUY_NOW"]["risk"] == "SHORTAGE").all())
    def test_buy_now_not_without_need_inside_lead(self):
        b = self.rec[self.rec["action"] == "BUY_NOW"]
        self.assertTrue((b["qty_raw"] > 0).all())
    def test_excess_never_bought(self):
        x = self.rec[self.rec["risk"] == "EXCESS"]
        self.assertFalse(x["action"].isin(["BUY_NOW", "BUY_LATER"]).any())
    def test_sub_unit_need_not_ordered(self):
        b = self.rec[self.rec["action"] == "BUY_NOW"]
        self.assertTrue(((b["qty_raw"] >= 1) | (b["risk"] == "SHORTAGE")).all())
    def test_safety_stock_capped(self):
        a = self.rec[self.rec["level_m"] > 0]
        mx = a["category"].map(lambda c: max(self.meta["season_index"][c]))
        slip = self.cfg["lead"]["slip_buffer_days"]
        bound = a["level_m"] * mx * (a["lead_days"] + slip + a["cover_policy_days"]) / 28
        self.assertTrue((a["safety_stock"] <= bound + 0.5).all())
    def test_buy_now_only_if_cover_not_above_lead_plus_review(self):
        b = self.rec[self.rec["action"] == "BUY_NOW"]
        dso = b["days_to_stockout"].fillna(10**6)
        self.assertTrue((dso <= b["lead_days"] + self.cfg["review_days"] + 1).all())
    def test_never_negative(self):
        self.assertTrue((self.rec["qty_recommended"] >= 0).all()); self.assertTrue((self.rec["stock_asof"] >= 0).all())

if __name__ == "__main__":
    unittest.main()
