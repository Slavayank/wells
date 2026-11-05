# CLAUDE.md — проект procure
Расчётный движок планирования закупок. **Числа считает код, не LLM.** LLM допускается только для чтения результатов и объяснений (`prompts/`).

## Команды
- `python procure.py run [--as-of YYYY-MM-DD] [--force]` — рекомендации → `runs/<id>/` (CSV + `dashboard.html` + `run_manifest.json`)
- `python procure.py backtest` / `sensitivity` — обновляют `runs/backtest_latest.csv`, `runs/sensitivity_latest.csv` (делать ДО `run`, дашборд их подхватывает)
- `python procure.py explain SKU-XXXXXX`
- `python -m unittest discover -s tests`  (pytest тоже подойдёт)

## Правила
- Все параметры только в `config.yaml`; не хардкодить пороги в коде. Допущения помечены ASSUMPTION.
- Одна дата среза `as_of` для всех таблиц; неполный месяц не входит в фит прогноза.
- Открытые заказы — только `purchase_orders.status=open`, `qty_pending` (не складывать с `sku_dim.on_order_qty`).
- Любое изменение формулы → обновить `docs/FORMULAS.md`, тест в `tests/`, прогнать `backtest` и сравнить с baseline (`seasonal_naive`).
- Не «улучшать» модель без backtest: цензурирование уже проверено и ухудшило результат.
- Рекомендации без `reason_codes` недопустимы.

## Структура
`src/procure/`: `config` · `data` (as-of, остатки, PO, лид-тайм, MOQ) · `forecast` · `engine` (формулы, статусы) · `backtest` · `report` (пакеты, календарь, дельта, sensitivity, дашборд) · `cli`.
`.claude/skills/`: `run-procurement-plan`, `explain-recommendation`.
