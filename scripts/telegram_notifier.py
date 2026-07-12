#!/usr/bin/env python3
"""
Deliver feature-drift Telegram alerts from the HOST.

Airflow (Docker) often cannot open HTTPS to api.telegram.org.
Monitoring queues messages into monitoring.alert_queue;
this script sends them using your normal host network.

  python scripts/telegram_notifier.py --once
  python scripts/telegram_notifier.py --loop --interval 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass


def db_url() -> str:
    host = os.getenv("APP_DB_HOST", "localhost")
    if host == "postgres":
        host = "localhost"  # host process talks to published port
    port = os.getenv("APP_DB_PORT", "5432")
    name = os.getenv("APP_DB_NAME", "credit_scoring")
    user = os.getenv("APP_DB_USER", "ml_user")
    password = os.getenv("APP_DB_PASSWORD", "ml_password")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def send(token: str, chat_id: str, text: str) -> tuple[bool, str]:
    import requests

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=30,
        )
        data = r.json() if r.content else {}
        if r.ok and data.get("ok"):
            return True, "ok"
        return False, str(data)
    except Exception as exc:
        return False, str(exc)


def drain() -> int:
    from sqlalchemy import create_engine, text

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return 1

    engine = create_engine(db_url(), pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS monitoring.alert_queue (
                    alert_id BIGSERIAL PRIMARY KEY,
                    message TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW(),
                    sent_at TIMESTAMP,
                    error TEXT
                )
                """
            )
        )

    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                    SELECT alert_id, message FROM monitoring.alert_queue
                    WHERE status = 'pending' ORDER BY created_at LIMIT 50
                    """
                )
            )
            .mappings()
            .all()
        )

    if not rows:
        print("No pending drift alerts")
        return 0

    for row in rows:
        ok, err = send(token, chat_id, row["message"])
        with engine.begin() as conn:
            if ok:
                conn.execute(
                    text(
                        "UPDATE monitoring.alert_queue "
                        "SET status='sent', sent_at=NOW(), error=NULL "
                        "WHERE alert_id=:id"
                    ),
                    {"id": row["alert_id"]},
                )
                print(f"sent #{row['alert_id']}")
            else:
                conn.execute(
                    text(
                        "UPDATE monitoring.alert_queue "
                        "SET status='error', error=:e WHERE alert_id=:id"
                    ),
                    {"id": row["alert_id"], "e": err[:500]},
                )
                print(f"fail #{row['alert_id']}: {err}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--once", action="store_true", default=True)
    args = p.parse_args()
    if args.loop:
        print(f"Polling every {args.interval}s (Ctrl+C to stop)")
        while True:
            drain()
            time.sleep(args.interval)
    return drain()


if __name__ == "__main__":
    raise SystemExit(main())
