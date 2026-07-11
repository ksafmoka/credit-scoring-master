.PHONY: up down restart logs shell-airflow shell-postgres init-db

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose down && docker-compose up -d

logs:
	docker-compose logs -f

shell-airflow:
	docker-compose exec airflow-scheduler bash

shell-postgres:
	docker-compose exec postgres psql -U postgres

init-db:
	docker-compose exec postgres psql -U postgres -f /docker-entrypoint-initdb.d/01_init_schema.sql

test:
	docker-compose exec airflow-scheduler pytest /opt/airflow/tests/ -v

format:
	black src/ dags/ tests/
	isort src/ dags/ tests/

lint:
	flake8 src/ dags/ tests/
	mypy src/