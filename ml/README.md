# ML Pipeline

Окремий пакет для навчання моделей. Поза backend, щоб не тягнути dev-залежності
(jupyter, optuna, shap, matplotlib) у production-образ.

## Структура

```
data_collection/  Збір реальних GitHub Actions runs через REST API
synthetic/        Синтетичний генератор CI runs з контрольованими розподілами
features/         Екстракція ознак (FR-04)
training/         Тренування моделей (RandomForest + LightGBM)
notebooks/        EDA, експерименти, аналіз результатів
artifacts/        (gitignored) серіалізовані моделі joblib
```

## Workflow

1. `python -m data_collection.collect --repos repos.txt --out data/raw/gha_runs.parquet`
2. `python -m synthetic.generate --n 10000 --out data/raw/synthetic.parquet`
3. `python -m features.extract --in data/raw --out data/processed/features.parquet`
4. `python -m training.train --features data/processed/features.parquet --out artifacts/`
