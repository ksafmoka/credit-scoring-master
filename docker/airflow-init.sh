#!/usr/bin/env bash
# Idempotent Airflow DB init + admin user bootstrap.
set -euo pipefail

echo "===== Airflow init start ====="

# Wait for Postgres (airflow metadata DB)
echo "Waiting for Postgres..."
for i in $(seq 1 60); do
  if airflow db check 2>/dev/null; then
    echo "Database is reachable."
    break
  fi
  if [[ "$i" -eq 60 ]]; then
    echo "ERROR: database not reachable after 60 attempts" >&2
    exit 1
  fi
  sleep 2
done

# Prefer migrate (Airflow >= 2.7); fall back to init for first boot
echo "Migrating / initializing metadata DB..."
if airflow db migrate; then
  echo "db migrate OK"
else
  echo "db migrate failed, trying db init..."
  airflow db init
fi

ADMIN_USER="${_AIRFLOW_WWW_USER_USERNAME:-admin}"
ADMIN_PASS="${_AIRFLOW_WWW_USER_PASSWORD:-admin}"
ADMIN_EMAIL="${_AIRFLOW_WWW_USER_EMAIL:-admin@example.com}"

echo "Ensuring admin user '${ADMIN_USER}' exists with known password..."

# If user already exists, reset password (handles re-init / leftover volume)
if airflow users list 2>/dev/null | awk '{print $1}' | grep -qx "${ADMIN_USER}"; then
  echo "User '${ADMIN_USER}' already exists — resetting password."
  airflow users reset-password \
    --username "${ADMIN_USER}" \
    --password "${ADMIN_PASS}"
else
  echo "Creating user '${ADMIN_USER}'..."
  airflow users create \
    --username "${ADMIN_USER}" \
    --password "${ADMIN_PASS}" \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email "${ADMIN_EMAIL}"
fi

echo "Current users:"
airflow users list || true

echo "===== Airflow init done. Login: ${ADMIN_USER} / ${ADMIN_PASS} ====="
