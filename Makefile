.PHONY: up down down-v rebuild restart-code airflow-admin sample-data local-pipeline test smoke prepare-lc telegram-once telegram-loop

# Start using EXISTING images (no rebuild) — preferred when only src/dags changed
up:
	docker compose up -d

# Full rebuild (needs stable PyPI network; can take long)
rebuild:
	docker compose build --no-cache
	docker compose up -d

# Recreate containers without rebuilding images (pick up compose/env)
restart-code:
	docker compose up -d --force-recreate

down:
	docker compose down

down-v:
	docker compose down -v

airflow-admin:
	docker compose run --rm --entrypoint /bin/bash airflow-init /opt/airflow/airflow-init.sh

sample-data:
	python scripts/generate_sample_data.py --n 5000

# Feature-drift Telegram delivery from HOST (Docker cannot reach Telegram)
telegram-once:
	PYTHONPATH=. python scripts/telegram_notifier.py --once

telegram-loop:
	PYTHONPATH=. python scripts/telegram_notifier.py --loop --interval 30

# make prepare-lc INPUT=/path/to/accepted.csv MAX_ROWS=200000
prepare-lc:
	PYTHONPATH=. python scripts/prepare_lending_club.py --input "$(INPUT)" --max-rows $(or $(MAX_ROWS),200000)

local-pipeline:
	PYTHONPATH=. python scripts/run_local_pipeline.py

test:
	PYTHONPATH=. pytest tests/ -v

smoke:
	PYTHONPATH=. python scripts/generate_sample_data.py --n 3000
	PYTHONPATH=. python scripts/run_local_pipeline.py
	PYTHONPATH=. pytest tests/ -q
