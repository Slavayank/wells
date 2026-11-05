---
name: run-procurement-plan
description: Запустить расчёт закупок (procure run), проверить качество данных и кратко доложить, что изменилось с прошлого запуска. Использовать при «обнови рекомендации», «что пора докупить», «пересчитай закупки».
---
1. Если обновились CSV: `python procure.py backtest` и `sensitivity`, затем `python procure.py run`. Если CLI пишет «пересчёт пропущен» — данные не менялись; скажи это, не используй `--force` без просьбы.
2. Открой `runs/LATEST` → `run_manifest.json`: проверь `freshness`, `counts`, `prev_run`.
3. Доложи: сколько SKU BUY_NOW / риск дефицита / избыток; топ-10 по `priority`; содержимое `delta.csv`; флаги STALE_PO, STOCK_UNVERIFIED, COST_MISSING.
4. Не пересчитывай числа сам и не меняй рекомендации — только цитируй файлы. Если backtest-метрики хуже baseline, скажи об этом явно.
