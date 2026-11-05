---
name: explain-recommendation
description: Объяснить закупщику, почему по SKU выдана рекомендация (qty, дата заказа, статус). Использовать при «почему заказывать X», «откуда эта цифра».
---
Выполни `python procure.py explain <SKU>` и объясни по шагам: остаток (snapshot − продажи + приходы), pipeline и его политика, L и C, прогноз за L и за [L, L+C], SS, target, qty_raw → qty с партионностью, дата заказа. Назови `reason_codes` и `flags` простым языком. Используй только значения из вывода; формулы — `docs/FORMULAS.md`.
