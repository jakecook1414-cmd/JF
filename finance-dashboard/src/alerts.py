"""
Alerts utilities for price/percent move triggers.
"""
import csv
import json
import os
from datetime import datetime


ALERT_FIELDS = ["id", "ticker", "type", "value", "enabled", "cooldown_minutes", "last_triggered"]
LOG_FIELDS = [
    "timestamp",
    "id",
    "ticker",
    "type",
    "alert_type",
    "value",
    "price",
    "day_pct_move",
    "message",
    "details_json",
]


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _parse_iso(ts):
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def load_alerts(path="data/alerts.json"):
    _ensure_parent(path)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def save_alerts(alerts, path="data/alerts.json"):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)


def evaluate_alerts(alerts, latest_prices, day_pct_moves, now_utc):
    updated = []
    triggered = []

    for alert in alerts:
        a = dict(alert or {})
        a.setdefault("enabled", True)
        a.setdefault("cooldown_minutes", 0)
        a.setdefault("last_triggered", "")
        a["ticker"] = str(a.get("ticker", "")).upper()

        if not a["ticker"] or not a.get("type"):
            updated.append(a)
            continue

        if not a.get("enabled", True):
            updated.append(a)
            continue

        cooldown = 0
        try:
            cooldown = int(a.get("cooldown_minutes") or 0)
        except Exception:
            cooldown = 0

        last_ts = _parse_iso(a.get("last_triggered"))
        if last_ts and cooldown > 0:
            delta = (now_utc - last_ts).total_seconds()
            if delta < cooldown * 60:
                updated.append(a)
                continue

        latest = latest_prices.get(a["ticker"])
        day_move = day_pct_moves.get(a["ticker"])

        try:
            value = float(a.get("value"))
        except Exception:
            updated.append(a)
            continue

        triggered_now = False
        message = ""

        if a["type"] == "price_above" and latest is not None:
            if latest > value:
                triggered_now = True
                message = f"{a['ticker']} > {value}"
        elif a["type"] == "price_below" and latest is not None:
            if latest < value:
                triggered_now = True
                message = f"{a['ticker']} < {value}"
        elif a["type"] == "pct_move_day" and day_move is not None:
            if abs(day_move) >= value:
                triggered_now = True
                message = f"{a['ticker']} daily move >= {value}%"

        if triggered_now:
            ts = now_utc.isoformat()
            a["last_triggered"] = ts
            triggered.append({
                "timestamp": ts,
                "id": a.get("id", ""),
                "ticker": a["ticker"],
                "type": a["type"],
                "alert_type": a["type"],
                "value": value,
                "price": latest,
                "day_pct_move": day_move,
                "message": message,
                "details_json": "",
            })

        updated.append(a)

    return updated, triggered


def append_alert_log(triggered_events, path="data/alerts_log.csv"):
    if not triggered_events:
        return
    _ensure_parent(path)
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        for event in triggered_events:
            row = {k: event.get(k, "") for k in LOG_FIELDS}
            writer.writerow(row)


def _load_scanner_state(path):
    _ensure_parent(path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_scanner_state(state, path):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def scanner_can_trigger(ticker, expiry, now_utc, cooldown_hours=6, path="data/scanner_cooldown.json"):
    key = f"{ticker}|{expiry}"
    state = _load_scanner_state(path)
    last = _parse_iso(state.get(key))
    if last:
        delta = (now_utc - last).total_seconds()
        if delta < cooldown_hours * 3600:
            return False
    state[key] = now_utc.isoformat()
    _save_scanner_state(state, path)
    return True
