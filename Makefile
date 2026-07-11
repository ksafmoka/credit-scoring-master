.PHONY: up down restart logs shell-airflow shell-postgres init-db sample-data local-pipeline test format lint

up:
	docker compose up -d --build

down:
	docker compose down

restart:
	docker compose down && docker compose up -d --build

logs:
	docker compose logs -f

shell-airflow:
	docker compose exec airflow-scheduler bash

shell-postgres:
	docker compose exec postgres psql -U postgres

init-db:
	@echo "Schemas are applied automatically on first postgres start via sql/"
	@echo "If volume already exists, re-run: docker compose down -v && make up"

sample-data:
	python scripts/generate_sample_data.py --n 5000

local-pipeline:
	PYTHONPATH=. python scripts/run_local_pipeline.py

test:
	PYTHONPATH=. pytest tests/ -v

# One-shot local check after clone
smoke:
	PYTHONPATH=. python scripts/generate_sample_data.py --n 3000
	PYTHONPATH=. python scripts/run_local_pipeline.py
	PYTHONPATH=. pytest tests/ -q

test-docker:
	docker compose exec airflow-scheduler pytest /opt/airflow/tests/ -v

format:
	black src/ dags/ tests/ scripts/
	isort src/ dags/ tests/ scripts/

lint:
	ruff check src/ dags/ tests/ scripts/ || true
