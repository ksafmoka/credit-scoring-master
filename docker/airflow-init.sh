#!/usr/bin/env bash
# Idempotent Airflow DB init + admin user bootstrap.
set -euo pipefail

echo "===== Airflow init start ====="

echo "Waiting for Postgres pg_isready..."
for i in $(seq 1 90); do
  if pg_isready -h postgres -p 5432 -U postgres >/dev/null 2>&1; then
    echo "pg_isready OK (postgres user)"
    break
  fi
  if [[ "$i" -eq 90 ]]; then
    echo "ERROR: pg_isready failed after 90 attempts" >&2
    exit 1
  fi
  sleep 2
done

echo "Waiting for airflow database via psql..."
for i in $(seq 1 90); do
  if PGPASSWORD=airflow psql -h postgres -U airflow -d airflow -c "SELECT 1" >/dev/null 2>&1; then
    echo "airflow DB reachable via airflow user"
    break
  fi
  if PGPASSWORD=postgres psql -h postgres -U postgres -d airflow -c "SELECT 1" >/dev/null 2>&1; then
    echo "airflow DB exists (postgres superuser)"
    break
  fi
  echo "  attempt $i: airflow DB not ready yet..."
  sleep 2
done

# NOTE: airflow db check can fail on first boot when tables don't exist yet,
# so we don't fail hard here — we try migrate directly with retries.
echo "Attempting airflow db migrate (with retries)..."

for i in $(seq 1 20); do
  if airflow db migrate; then
    echo "db migrate OK on attempt $i"
    break
  fi
  echo "  migrate attempt $i failed, retrying in 5s..."
  if [[ "$i" -eq 20 ]]; then
    echo "migrate failed 20 times, trying db init as fallback..."
    if airflow db init; then
      echo "db init OK"
      break
    else
      echo "ERROR: both migrate and init failed" >&2
      exit 1
    fi
  fi
  sleep 5
done

ADMIN_USER="${_AIRFLOW_WWW_USER_USERNAME:-admin}"
ADMIN_PASS="${_AIRFLOW_WWW_USER_PASSWORD:-admin}"
ADMIN_EMAIL="${_AIRFLOW_WWW_USER_EMAIL:-admin@example.com}"

echo "Ensuring admin user '${ADMIN_USER}' exists..."

if airflow users list 2>/dev/null | awk '{print $1}' | grep -qx "${ADMIN_USER}"; then
  echo "User '${ADMIN_USER}' already exists — resetting password."
  airflow users reset-password \
    --username "${ADMIN_USER}" \
    --password "${ADMIN_PASS}" || true
else
  echo "Creating user '${ADMIN_USER}'..."
  airflow users create \
    --username "${ADMIN_USER}" \
    --password "${ADMIN_PASS}" \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email "${ADMIN_EMAIL}" || {
      echo "Create failed, trying reset..."
      airflow users reset-password --username "${ADMIN_USER}" --password "${ADMIN_PASS}" || true
    }
fi

echo "Current users:"
airflow users list || true

echo "===== Airflow init done. Login: ${ADMIN_USER} / ${ADMIN_PASS} ====="
