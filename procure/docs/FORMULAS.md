# Спецификация расчёта (источник истины — `engine.py`, `forecast.py`, `data.py`)

Все параметры в `config.yaml`. Пометка **ASSUMPTION** — допущение, не следующее из данных.

## 0. Единая дата среза (`as_of`, по умолчанию 2026-08-13)
Одна дата для всех таблиц. Правило roll-forward:
`stock(as_of) = max(0, snapshot(1-е число месяца as_of) − net_sales(1..as_of) + receipts(1..as_of))`.
- Неполный месяц используется **только** в roll-forward, в фит прогноза не входит (`P.months` заканчивается на последнем полном месяце).
- Если `as_of` — 1-е число, продажи неполного месяца не используются.
- Нулевых строк в `stock_monthly` нет (проверено). Отсутствие строки = 0, статус: `row_absent_zero` (подтверждается динамикой: roll-forward даёт ≤0 в 80% случаев) или `never_snapshotted` (не верифицировано → ручная проверка, если есть спрос).

## 1. Спрос
`net = max(0, qty_sold − qty_returned)` (возврат возвращается на склад: roll-forward сходится точнее — 63.6% vs 58.5% точных совпадений).
Сезонный индекс категории: отношение ряда к центрированной скользящей 2×12, медиана по годам, mean=1.
Уровень SKU: `level = Σ w·net / Σ w·seas` по валидным месяцам окна (24 мес., период полураспада 6 мес.).
Цензурирование (`use_censoring`): месяцы со `stock=0` на 1-е и `sales=0` исключаются из фита. **Выключено по умолчанию**: backtest показал смещение +26% против +17% (исключение создаёт selection bias). Флаг `DEMAND_LOWER_BOUND` ставится, если валидных месяцев < `min_obs`.
`demand_scale` (по умолчанию 1.0): ручная поправка к смещению модели (в backtest ~+17%).

## 2. Плечо L и покрытие C
`L = clip(median(planned_lead_days SKU за 36 мес.) → поставщик → supplier_dim → default, 14, 180)`.
`C` — политика покрытия по категории (**ASSUMPTION**, в ERP `target_cover_days = 0` у всех SKU).

## 3. Safety stock
`SS = z(service_level) · σ_u · √( Σ_{t=1}^{L+slip} idx_t / dim_t )`
где `σ_u` — RMS остатков `(net − level·seas)/√seas` (сезонно-нормированный), `slip` = 14 дн (**ASSUMPTION**).

## 4. Проекция и заказ
`P(t) = s0 + Σ приходов(≤t) − Σ спроса(≤t)`, t = 1..540 дней.
- `target = Σ спроса[L, L+C] + SS`
- `qty_raw = max(0, target − max(0, P(L)))`
- `qty = ceil(max(qty_raw, min_batch)/lot)·lot` (MOQ-прокси из истории PO: `min_batch` = p25 прошлых заказов SKU, `lot` — наибольшая из {100,50,10,5,1}, кратная ≥80% заказов)
- `days_to_stockout` = первый день, когда `P(t) ≤ −1` (ожидаемый неудовлетворённый спрос ≥ 1 шт)
- `t_ss` = первый день **t ≥ L**, когда `P(t) ≤ SS` (заказ, размещённый сегодня, приходит только на день L, поэтому провалы до L он исправить не может); **`next_order_date = as_of + (t_ss − L)`**, не меньше 0
- `cover_with_pipeline_days` = `days_to_stockout` (симуляция); `cover_formula_days = (stock + pipeline_eff)/avg_daily_forecast(L+C)`.

## 5. Открытые PO (единственный источник — `purchase_orders.status=open`, `qty_pending`; `sku_dim.on_order_qty` не используется — иначе дубль)
| состояние | условие | вес | приход |
|---|---|---|---|
| on_time | promised ≥ as_of | надёжность поставщика (1.0, если `qty_confirmed ≥ pending`) | promised |
| late | просрочка ≤ 60 дн | надёжность × 0.5 | as_of + 21 дн |
| stale | просрочка > 60 дн | 0, флаг `STALE_PO` | не учитывается |
Надёжность = доля исполнения по терминальным PO (completed/closed), сжатая к среднему (k=5), пол 0.3.

## 6. Статусы (приоритет сверху вниз)
1. `DEAD_STOCK`: нет продаж `dead_months`=12 и остаток/приходы > 0; `INACTIVE`: нет продаж и нет остатка.
2. `MANUAL_CHECK`: продавался за 12 мес., но нет себестоимости (`COST_MISSING`) — **не** мёртвый ассортимент; либо остаток не верифицирован.
3. `REVIEW_ASSORTMENT`: наценка < `MIN_MARKUP_PCT`=25 (не попадает в авто-закупку).
4. `MANUAL_CHECK/MOQ_DOMINATES`: мин. партия > 3× потребности (`moq.dominance_ratio`) и нет риска дефицита — не заказывать автоматически.
5. `BUY_NOW`: `qty>0` и `next_order_in_days ≤ review_days`(30); `BUY_LATER` — позже; `NO_ACTION` — покрыто.
Риск (независимо): `SHORTAGE` — `days_to_stockout < L`; `EXCESS` — покрытие (симуляция **и** формула) `> L + C + 60`.

## 7. Приоритет (денег в данных нет)
`priority = urgency · (0.5 + 0.1·price_decile) · ln(1 + exposure)`, `urgency = clip(1 − days_to_stockout/(L+30), 0, 1)`,
`exposure = недопокрытый спрос за плечо + недопокрытый спрос за период покрытия`. Нормируется к 0–100.
