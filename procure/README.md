# procure — движок планирования закупок
```
pip install -r requirements.txt
python procure.py backtest && python procure.py sensitivity
python procure.py run            # -> runs/<id>/dashboard.html
python -m unittest discover -s tests
```
Данные: `data/raw/*.csv` (путь в `config.yaml`). Формулы: `docs/FORMULAS.md`. Решения: `docs/DECISIONS.md`. Для CLAUDE/агентов: `CLAUDE.md`, `.claude/skills/`.
Повторный запуск: заменить CSV (или поменять `as_of`) и выполнить `run` — CLI сравнит с предыдущим запуском (`delta.csv`, вкладка «Что изменилось») и пропустит пересчёт, если ничего не изменилось.
