-- Runs once on empty Postgres volume (as superuser `postgres`).
-- Creates app roles, databases, schemas owned by ml_user.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ml_user') THEN
        CREATE USER ml_user WITH PASSWORD 'ml_password';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'airflow') THEN
        CREATE USER airflow WITH PASSWORD 'airflow';
    END IF;
END
$$;

SELECT 'CREATE DATABASE credit_scoring OWNER ml_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'credit_scoring')\gexec

SELECT 'CREATE DATABASE airflow OWNER airflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

SELECT 'CREATE DATABASE mlflow OWNER postgres'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

GRANT ALL PRIVILEGES ON DATABASE credit_scoring TO ml_user;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;

\c credit_scoring

CREATE SCHEMA IF NOT EXISTS raw AUTHORIZATION ml_user;
CREATE SCHEMA IF NOT EXISTS features AUTHORIZATION ml_user;
CREATE SCHEMA IF NOT EXISTS models AUTHORIZATION ml_user;
CREATE SCHEMA IF NOT EXISTS predictions AUTHORIZATION ml_user;
CREATE SCHEMA IF NOT EXISTS monitoring AUTHORIZATION ml_user;

-- Allow ml_user full control inside its schemas
GRANT ALL ON SCHEMA raw, features, models, predictions, monitoring TO ml_user;
ALTER ROLE ml_user SET search_path TO raw, features, predictions, monitoring, public;
