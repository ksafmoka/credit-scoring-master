# Data

## Lending Club (primary)

Place your dump here:

```text
data/lending_club.csv
```

Ingestion **automatically prefers this file** when it exists  
(`/opt/airflow/data/lending_club.csv` inside Docker).

Then re-run:

```text
data_ingestion → feature_engineering → model_training → batch_prediction
```

Optional helper to subsample a huge dump and suggest date splits:

```bash
python scripts/prepare_lending_club.py --input /path/to/full.csv --max-rows 200000
```

## Demo sample

```bash
python scripts/generate_sample_data.py --n 5000
# data/sample_applications.csv — used only if lending_club.csv is missing
```

## Env override

```env
LENDING_CLUB_CSV_PATH=/opt/airflow/data/lending_club.csv
```
