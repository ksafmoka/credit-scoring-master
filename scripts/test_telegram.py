#!/usr/bin/env python3
"""Send a test Telegram message using TELEGRAM_* env vars."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass


def main() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print(
            "ERROR: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID "
            "(e.g. in .env or export them)."
        )
        return 1

    # Prefer project helper if importable
    try:
        from src.monitoring.data_drift import send_telegram_alert

        ok = send_telegram_alert(
            "✅ Credit Scoring monitoring: Telegram test message.\n"
            "If you see this, alerts are configured correctly."
        )
        print("OK" if ok else "FAILED (see logs above)")
        return 0 if ok else 2
    except Exception:
        import requests

        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "✅ Credit Scoring monitoring: Telegram test message.",
            },
            timeout=15,
        )
        print(resp.status_code, resp.text)
        return 0 if resp.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
