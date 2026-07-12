# Train on real Lending Club data

## 1. Get the CSV

Download **Lending Club accepted loans** (Kaggle “Lending Club Loan Data” / accepted loans file).

Save as:

```text
data/lending_club.csv
```

(`./data` is mounted into Airflow as `/opt/airflow/data`)

Optional: keep only needed years to reduce size (example with Python on host):

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/lending_club.csv", low_memory=False)
# keep recent years if issue_d is like "Dec-2015"
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
df = df[df["issue_d"] >= "2015-01-01"]
df.to_csv("data/lending_club.csv", index=False)
print(len(df), df["issue_d"].min(), df["issue_d"].max())
PY
```

## 2. Point Airflow to that file

In project `.env`:

```env
LENDING_CLUB_CSV_PATH=/opt/airflow/data/lending_club.csv
```

Or edit `docker-compose.yml` → `LENDING_CLUB_CSV_PATH` the same way.

Recreate scheduler/webserver:

```bash
docker compose up -d --force-recreate airflow-scheduler airflow-webserver
```

## 3. Align train/val/test dates with your data

Check date range:

```bash
# host
python -c "import pandas as pd; d=pd.read_csv('data/lending_club.csv', usecols=['issue_d']); print(d['issue_d'].dropna().head()); print(d['issue_d'].value_counts().head())"
```

Edit `src/config.py` → `TrainingConfig` if needed, e.g. for older LC dumps:

```python
TRAIN_END_DATE = "2016-12-31"
VAL_END_DATE = "2017-06-30"
```

(Defaults `2022-12-31` / `2023-06-30` only fit modern samples.)

## 4. Re-run pipeline

```text
data_ingestion → feature_engineering → model_training → batch_prediction
```

```bash
docker compose exec airflow-scheduler airflow dags trigger data_ingestion
# wait SUCCESS, then:
docker compose exec airflow-scheduler airflow dags trigger feature_engineering
docker compose exec airflow-scheduler airflow dags trigger model_training
docker compose exec airflow-scheduler airflow dags trigger batch_prediction
curl -X POST http://localhost:8000/reload
```

`data_ingestion` truncates raw tables and reloads from the CSV path.

## 5. Column mapping

Loader maps common LC names (`loan_amnt`, `annual_inc`, `loan_status`, `issue_d`, …) in `src/data/ingestion.py` → `COLUMN_MAP`.  
Add aliases there if your dump uses different headers.

## 6. Practical tips

- Full LC is huge → start with 100k–500k rows  
- Training with Optuna will take much longer than on 4k sample  
- Synthetic payment/bureau history is still generated if you use the default ingestion helpers (not real LC payments)
