#!/usr/bin/env python3
"""
Recover missed trade records from bot log files.

Bug: live exit path didn't call _log_trade, so EXIT_TP/EXIT_SL events
were never written to late_game_trades.log or the database.

This script parses all log files, extracts every ENTRY and EXIT event
(with PnL), and regenerates the trade log.

Usage:
    python scripts/recover_trades.py                  # preview only
    python scripts/recover_trades.py --write-log      # append to late_game_trades.log
    python scripts/recover_trades.py --write-db       # also write to database
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
TRADE_LOG_PATH = PROJECT_ROOT / "late_game_trades.log"

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Label always looks like: [Team Name (MM-DD HH:MM)]
# Match a capital letter start, anything, then date-time in parens
LABEL_RE = r"\[(?P<label>[A-Z][^\]]*?\(\d{2}-\d{2} \d{2}:\d{2}\))\]"

# ENTRY (always has shares + cost):
#   [Label] ENTRY — price=X shares=Y cost=$Z
ENTRY_RE = re.compile(
    LABEL_RE + r"\s*ENTRY\s*[-–—]\s*"
    r"price=(?P<price>[0-9.]+)\s+shares=(?P<shares>[0-9.]+)\s+cost=\$(?P<cost>[0-9.]+)"
)

# EXIT v2 (newer — has shares + sell):
#   [Label] Take-profit +X% — price=X shares=Y sell=Z pnl=$W
EXIT_V2_RE = re.compile(
    LABEL_RE + r"\s*(?P<reason>Take-profit|Stop-loss)\s+[+-][0-9.]+\s*%\s*[-–—]\s*"
    r"price=(?P<price>[0-9.]+)\s+shares=(?P<shares>[0-9.]+)\s+sell=(?P<sell>[0-9.]+)\s+pnl=\$(?P<pnl>[+-]?[0-9.]+)"
)

# EXIT v1 (older — no shares/sell, just price + pnl):
#   [Label] Stop-loss -X% — price=X pnl=$Y
EXIT_V1_RE = re.compile(
    LABEL_RE + r"\s*(?P<reason>Take-profit|Stop-loss)\s+[+-][0-9.]+\s*%\s*[-–—]\s*"
    r"price=(?P<price>[0-9.]+)\s+pnl=\$(?P<pnl>[+-]?[0-9.]+)"
)

# Timestamp at beginning of log line
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_logs() -> list[dict]:
    """Parse all log files and return chronological list of trade events."""
    events: list[dict] = []
    log_files = sorted(LOGS_DIR.glob("trading_system_*.log"))

    for log_path in log_files:
        try:
            content = log_path.read_text(encoding="utf-8")
        except Exception:
            continue

        for line in content.splitlines():
            clean = strip_ansi(line)

            # Extract timestamp
            ts_match = TS_RE.match(clean)
            ts = None
            if ts_match:
                try:
                    ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass

            # Try EXIT v2 first (more specific — has shares + sell)
            m = EXIT_V2_RE.search(clean)
            if m:
                reason_map = {"Take-profit": "EXIT_TP", "Stop-loss": "EXIT_SL"}
                events.append({
                    "type": reason_map.get(m.group("reason"), "EXIT"),
                    "label": m.group("label").strip(),
                    "price": float(m.group("price")),
                    "shares": float(m.group("shares")),
                    "sell": int(float(m.group("sell"))),
                    "pnl": float(m.group("pnl")),
                    "timestamp": ts,
                    "source": log_path.name,
                })
                continue

            # Try EXIT v1 (older — no shares/sell)
            m = EXIT_V1_RE.search(clean)
            if m:
                reason_map = {"Take-profit": "EXIT_TP", "Stop-loss": "EXIT_SL"}
                events.append({
                    "type": reason_map.get(m.group("reason"), "EXIT"),
                    "label": m.group("label").strip(),
                    "price": float(m.group("price")),
                    "shares": 0,  # unknown in v1 logs
                    "sell": 0,
                    "pnl": float(m.group("pnl")),
                    "timestamp": ts,
                    "source": log_path.name,
                })
                continue

            # Try ENTRY
            m = ENTRY_RE.search(clean)
            if m:
                events.append({
                    "type": "ENTRY",
                    "label": m.group("label").strip(),
                    "price": float(m.group("price")),
                    "shares": float(m.group("shares")),
                    "cost": float(m.group("cost")),
                    "pnl": 0.0,
                    "timestamp": ts,
                    "source": log_path.name,
                })

    events.sort(key=lambda e: e["timestamp"] or datetime.min.replace(tzinfo=timezone.utc))
    return events


def format_log_line(ev: dict) -> str:
    """Format an event as a late_game_trades.log line."""
    ts = ev["timestamp"]
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "unknown"

    label = ev["label"][:30]
    action = ev["type"]
    price = ev["price"]
    shares = int(ev.get("sell", ev.get("shares", 0)))
    if shares == 0:
        shares = int(ev.get("shares", 0))
    pnl = ev.get("pnl", 0.0)

    # pnl_pct: compute from price if we have shares and entry price
    pnl_pct = 0.0
    entry_price = ev.get("entry_price", 0.0)
    if entry_price > 0 and shares > 0:
        cost = entry_price * shares
        if cost > 0:
            pnl_pct = (pnl / cost) * 100

    return (
        f"[{ts_str}] {action:<8} | {label:<30} | {'?':>4} "
        f"@ {price:.4f} x {shares} | PnL=${pnl:+.2f} ({pnl_pct:+.1f}%)"
        f" | "
    )


def main():
    parser = argparse.ArgumentParser(description="Recover missed trade records from logs")
    parser.add_argument("--write-log", action="store_true", help="Append recovered exits to late_game_trades.log")
    parser.add_argument("--write-db", action="store_true", help="Write to database (requires src)")
    parser.add_argument("--all", action="store_true", help="Also recover entries (not just exits)")
    args = parser.parse_args()

    events = parse_logs()

    # Separate entries and exits
    entries = [e for e in events if e["type"] == "ENTRY"]
    exits = [e for e in events if e["type"] in ("EXIT_TP", "EXIT_SL")]

    print(f"Found {len(events)} total events across log files:")
    print(f"  ENTRY:    {len(entries)}")
    print(f"  EXIT_TP:  {sum(1 for e in exits if e['type'] == 'EXIT_TP')}")
    print(f"  EXIT_SL:  {sum(1 for e in exits if e['type'] == 'EXIT_SL')}")
    print()

    # Summary
    total_pnl = sum(e["pnl"] for e in exits)
    wins = sum(1 for e in exits if e["pnl"] > 0)
    losses = sum(1 for e in exits if e["pnl"] <= 0)
    print(f"Exit summary: {len(exits)} exits | {wins} wins / {losses} losses | PnL=${total_pnl:+.2f}")
    print()

    # Show exits by date
    by_date: dict[str, list[dict]] = {}
    for e in exits:
        date_key = e["timestamp"].strftime("%Y-%m-%d") if e["timestamp"] else "unknown"
        by_date.setdefault(date_key, []).append(e)

    for date_key in sorted(by_date):
        day_exits = by_date[date_key]
        day_pnl = sum(e["pnl"] for e in day_exits)
        day_wins = sum(1 for e in day_exits if e["pnl"] > 0)
        day_losses = sum(1 for e in day_exits if e["pnl"] <= 0)
        print(f"  {date_key}: {len(day_exits):2d} exits | {day_wins}W/{day_losses}L | PnL=${day_pnl:+.2f}")

    print()

    # Show detailed exits
    print("--- All exit events ---")
    for e in exits:
        ts = e["timestamp"].strftime("%m-%d %H:%M:%S") if e["timestamp"] else "?"
        emoji = "🟢" if e["pnl"] > 0 else "🔴"
        shares_str = f"x {e['sell']}" if e.get("sell") else f"x {e.get('shares', '?')}"
        print(f"  {emoji} {ts} | {e['label']:<30} | {e['type']:<8} | "
              f"@ {e['price']:.4f} {shares_str} | PnL=${e['pnl']:+.2f}")

    # Write to log file
    if args.write_log:
        existing = set()
        if TRADE_LOG_PATH.exists():
            for line in TRADE_LOG_PATH.read_text().splitlines():
                existing.add(line.strip()[:80])

        new_count = 0
        with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
            for e in exits:
                line = format_log_line(e)
                fp = line.strip()[:80]
                if fp not in existing:
                    f.write(line + "\n")
                    new_count += 1
            if args.all:
                for e in entries:
                    line = format_log_line(e)
                    fp = line.strip()[:80]
                    if fp not in existing:
                        f.write(line + "\n")
                        new_count += 1

        print(f"\nAppended {new_count} new lines to {TRADE_LOG_PATH}")

    # Write to database
    if args.write_db:
        asyncio.run(_write_db(events))


async def _write_db(events: list[dict]):
    """Write recovered exits to the SQLite database."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.database import DatabaseManager, TradeLog

    db = DatabaseManager()
    await db.initialize()

    exits = [e for e in events if e["type"] in ("EXIT_TP", "EXIT_SL")]
    entries_by_label: dict[str, list[dict]] = {}
    for e in events:
        if e["type"] == "ENTRY":
            entries_by_label.setdefault(e["label"], []).append(e)

    added = 0
    for e in exits:
        label = e["label"]
        # Find closest entry before this exit
        entry_list = entries_by_label.get(label, [])
        entry_ts = e["timestamp"]
        entry_price = 0.0
        for ent in entry_list:
            if ent["timestamp"] and ent["timestamp"] < (e["timestamp"] or datetime.max.replace(tzinfo=timezone.utc)):
                entry_ts = ent["timestamp"]
                entry_price = ent["price"]

        trade = TradeLog(
            market_id="",
            side="?",
            entry_price=entry_price,
            exit_price=e["price"],
            quantity=e.get("sell", e.get("shares", 0)),
            pnl=e["pnl"],
            entry_timestamp=entry_ts,
            exit_timestamp=e["timestamp"],
            rationale=e["type"],
            strategy="late_game_favorite",
        )
        try:
            await db.add_trade_log(trade)
            added += 1
        except Exception as ex:
            print(f"  DB write error: {ex}")

    print(f"Wrote {added} trade records to database.")


if __name__ == "__main__":
    main()
