# Data

Place application CSVs here.

## Quick start (recommended for demo)

```bash
python scripts/generate_sample_data.py --n 5000
# writes data/sample_applications.csv
```

Or run the full local smoke pipeline (also writes the sample CSV + model artifacts):

```bash
make local-pipeline
```

## Real Lending Club

Download a Lending Club accepted-loans dump (Kaggle / original source) and save as:

```text
data/lending_club.csv
```

Then set in `.env`:

```env
LENDING_CLUB_CSV_PATH=/opt/airflow/data/lending_club.csv
```

The loader maps common Lending Club columns (`loan_amnt`, `annual_inc`, `loan_status`, …)
to the internal `raw.applications` schema.
