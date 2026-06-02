"""
MLB Late-Game Favorite Strategy — price-driven entry/exit for prediction markets.

Strategy:
  - Monitor both sides (YES/NO tokens) per MLB moneyline game independently.
  - Enter when token price crosses entry_confidence (0.80), game not over.
  - Exit on: take-profit (+80%), stop-loss (-13%), or game-over (>=0.99 or <=0.01).

In a Polymarket binary "Will X beat Y?" market:
  - YES token = bet on team X (first mentioned)
  - NO token  = bet on team Y (second mentioned)
We track each token's own midpoint price independently.
"""

from __future__ import annotations

import asyncio
import json
import re
import time as _time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from src.clients.gamma_client import GammaClient
from src.clients.polymarket_client import PolymarketClient
from src.utils.database import DatabaseManager, TradeLog
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("late_game")

# ---------------------------------------------------------------------------
# Market filtering
# ---------------------------------------------------------------------------

MLB_TEAM_KEYWORDS = [
    "dodgers", "yankees", "red sox", "cubs", "mets", "braves", "astros",
    "phillies", "cardinals", "giants", "padres", "tigers", "twins", "royals",
    "rangers", "athletics", "pirates", "nationals", "orioles", "rays",
    "mariners", "angels", "rockies", "brewers", "reds", "marlins", "white sox",
    "blue jays", "diamondbacks", "guardians",
]

SKIP_TITLE_KEYWORDS = [
    "over/under", "o/u", "total runs", "runs scored", "handicap", "spread",
    "run line", "world series", "pennant", "mvp", "championship",
    "division winner", "playoffs", "all-star", "home run", "strikeout",
    "hits", "rbi", "first inning", "first pitch", "national anthem",
]


def _parse_game_time(raw: str) -> float:
    """Parse a game start time string to a Unix timestamp. Returns 0 on failure."""
    if not raw:
        return 0.0
    try:
        s = raw.strip().replace(" ", "T", 1)
        if "+" in s:
            tz_part = s.rsplit("+", 1)[-1]
            if len(tz_part) == 2:
                s += ":00"
        elif s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _is_mlb_moneyline(title: str) -> bool:
    t = title.lower()
    if any(kw in t for kw in SKIP_TITLE_KEYWORDS):
        return False
    if not any(kw in t for kw in MLB_TEAM_KEYWORDS):
        return False
    return any(w in t for w in [" vs ", " vs. ", " v ", " @ ", " at ",
                                "to beat", "beats ", "will beat", "will win"])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LateGameFavoriteConfig:
    entry_confidence: float = 0.625
    game_over_high: float = 0.995
    game_over_low: float = 0.005
    take_profit: float = 1.00    # +80% from entry
    stop_loss: float = 0.10      # -10% from entry
    trade_size: float = 10     # shares per trade
    poll_interval: int = 3       # seconds between price polls
    max_games: int = 20          # max concurrent games to track


# ---------------------------------------------------------------------------
# TokenWatcher — per-token state machine
# ---------------------------------------------------------------------------

class TokenState(Enum):
    WATCHING = "watching"
    IN_POSITION = "in_position"
    CLOSED = "closed"


@dataclass
class TokenWatcher:
    token_id: str           # token to buy and monitor
    condition_id: str       # parent market
    side: str               # "yes" or "no"
    label: str              # e.g. "Dodgers (AWAY)"
    market_title: str
    neg_risk: bool = False
    state: TokenState = TokenState.WATCHING
    current_price: float = 0.0
    entry_price: float = 0.0
    peak_price: float = 0.0  # highest price since entry (trailing stop reference)
    shares: int = 0
    order_id: str = ""
    pnl: float = 0.0
    cooldown_scans: int = 0  # scans to wait before allowing re-entry

    def update_price(self, price: float, cfg: "LateGameFavoriteConfig") -> Optional[str]:
        self.current_price = price
        if self.cooldown_scans > 0:
            self.cooldown_scans -= 1

        if self.state == TokenState.WATCHING:
            if price <= cfg.game_over_low or price >= cfg.game_over_high:
                return "GAME_OVER"
            if self.cooldown_scans <= 0 and price >= cfg.entry_confidence:
                return "ENTRY"

        elif self.state == TokenState.IN_POSITION:
            if price <= cfg.game_over_low or price >= cfg.game_over_high:
                return "GAME_OVER"
            # Update trailing stop peak
            if price > self.peak_price:
                self.peak_price = price
            if self.entry_price > 0:
                # Take-profit: based on entry price
                pnl_pct = (price - self.entry_price) / self.entry_price
                if pnl_pct >= cfg.take_profit:
                    return "EXIT_TP"
                # Stop-loss: trailing — based on peak price
                if self.peak_price > 0:
                    drawdown = (price - self.peak_price) / self.peak_price
                    if drawdown <= -cfg.stop_loss:
                        return "EXIT_SL"

        return None


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class LateGameFavoriteStrategy:
    """MLB late-game favorite — multi-game, price-driven entry/exit."""

    def __init__(
        self,
        client: PolymarketClient,
        gamma: Optional[GammaClient] = None,
        config: Optional[LateGameFavoriteConfig] = None,
        dry_run: bool = True,
        db: Optional[DatabaseManager] = None,
    ):
        self.client = client
        self.gamma = gamma or GammaClient()
        self._owns_gamma = gamma is None
        self.config = config or LateGameFavoriteConfig()
        self.dry_run = dry_run
        self.watchers: Dict[str, TokenWatcher] = {}
        self._running = False
        self._db = db
        self._trade_log_path = Path(__file__).resolve().parent.parent.parent / "late_game_trades.log"
        self._open_trades: Dict[str, datetime] = {}  # token_id → entry time

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def discover_games(self) -> List[Dict[str, Any]]:
        """Find active MLB moneyline markets via Gamma API with tag_slug=mlb."""
        logger.info("Discovering MLB moneyline markets...")

        gam_host = self.gamma.host if hasattr(self.gamma, 'host') else "https://gamma-api.polymarket.com"
        all_markets: List[Dict[str, Any]] = []
        seen: set[str] = set()

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as sess:
            for page in range(5):
                params = {
                    "active": "true", "closed": "false", "limit": 200,
                    "offset": page * 200, "order": "volume", "ascending": "false",
                    "tag_slug": "mlb",
                }
                try:
                    r = await sess.get(f"{gam_host}/events", params=params)
                    if r.status_code != 200:
                        break
                    data = r.json()
                    if not isinstance(data, list) or not data:
                        break
                except Exception:
                    break

                for event in data:
                    for m in event.get("markets", []) or []:
                        cond = m.get("conditionId") or ""
                        title = m.get("question") or ""
                        if not cond or cond in seen:
                            continue
                        if not m.get("acceptingOrders", True):
                            continue
                        # Only moneyline: has vs/beat, no O/U or props
                        if not _is_mlb_moneyline(title):
                            continue
                        seen.add(cond)

                        raw_ids = m.get("clobTokenIds") or "[]"
                        try:
                            ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
                            yes_id = str(ids[0]) if len(ids) >= 1 else ""
                            no_id = str(ids[1]) if len(ids) >= 2 else ""
                        except (json.JSONDecodeError, TypeError):
                            yes_id, no_id = "", ""

                        if not cond or not yes_id:
                            continue

                        neg_risk = bool(m.get("negRisk", False))
                        tick_size = float(m.get("orderPriceMinTickSize", 0.01) or 0.01)
                        self.client.register_market(cond, yes_id, no_id,
                                                    neg_risk=neg_risk, tick_size=tick_size)

                        # Parse game start time for sorting (format: "2026-05-21 23:05:00+00")
                        game_time_raw = m.get("gameStartTime") or ""
                        game_ts = _parse_game_time(game_time_raw)

                        all_markets.append({
                            "condition_id": cond,
                            "title": title,
                            "yes_token": yes_id,
                            "no_token": no_id,
                            "neg_risk": neg_risk,
                            "volume": float(m.get("volumeNum") or m.get("volume") or 0),
                            "game_start": game_ts,
                        })

        # Sort by distance from now — in-progress and soon-starting first
        now = datetime.now(timezone.utc).timestamp()
        def _sort_key(m: dict) -> tuple:
            ts = m.get("game_start") or 0
            if ts == 0:
                return (2, 0)  # no time → end
            return (0, abs(ts - now))  # closest to now first

        all_markets.sort(key=_sort_key)
        all_markets = all_markets[:self.config.max_games]
        logger.info("Found %d MLB moneyline markets", len(all_markets))
        return all_markets

    @staticmethod
    def _extract_teams(title: str) -> tuple:
        patterns = [
            r"will\s+(.+?)\s+(?:win\s+(?:vs?\.?\s+|against\s+|over\s+)|beat\s+|defeat\s+)(.+?)(?:\?|$|\s+on|\s+\()",
            r"^(.+?)\s+vs\.?\s+(.+?)(?:\?|$|\s+on|\s+[-–\(])",
            r"^(.+?)\s+v\s+(.+?)(?:\?|$|\s+on|\s+[-–\(])",
            r"^(.+?)\s+(?:at|@)\s+(.+?)(?:\?|$|\s+on|\s+[-–\(])",
        ]
        for pat in patterns:
            m = re.search(pat, title, re.I)
            if m:
                return m.group(1).strip()[:40], m.group(2).strip()[:40]
        return "Team A", "Team B"

    # ------------------------------------------------------------------
    # Price fetch
    # ------------------------------------------------------------------

    async def _fetch_token_price(self, token_id: str) -> Optional[float]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as sess:
                r = await sess.get(
                    "https://clob.polymarket.com/midpoint",
                    params={"token_id": token_id, "_": int(_time.time() * 1000)},
                )
                if r.status_code == 200:
                    v = r.json().get("mid")
                    return float(v) if v is not None else None
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        cfg = self.config
        mode = "DRY RUN" if self.dry_run else "LIVE"

        print(f"\n{'='*60}")
        print(f"MLB LATE-GAME FAVORITE — {mode}")
        print(f"  Entry: price >= {cfg.entry_confidence}")
        print(f"  Exit:  TP +{cfg.take_profit*100:.0f}% | SL -{cfg.stop_loss*100:.0f}%")
        print(f"         Game-over >= {cfg.game_over_high} / <= {cfg.game_over_low}")
        print(f"  Size:  {cfg.trade_size:.0f} shares | Poll: {cfg.poll_interval}s")
        print(f"{'='*60}\n")

        games = await self.discover_games()
        if not games:
            print("No MLB moneyline markets found.")
            return

        self.watchers.clear()
        for g in games:
            away, home = self._extract_teams(g["title"])
            game_ts = g.get("game_start") or 0
            time_str = datetime.fromtimestamp(game_ts, tz=timezone.utc).strftime("%m-%d %H:%M") if game_ts else "no time"
            label_away = f"{away} ({time_str})"
            label_home = f"{home} ({time_str})"
            # YES token = bet on away team (first mentioned in "Will X beat Y?")
            self.watchers[g["yes_token"]] = TokenWatcher(
                token_id=g["yes_token"], condition_id=g["condition_id"],
                side="yes", label=label_away, market_title=g["title"],
                neg_risk=g["neg_risk"],
            )
            # NO token = bet on home team (second mentioned)
            self.watchers[g["no_token"]] = TokenWatcher(
                token_id=g["no_token"], condition_id=g["condition_id"],
                side="no", label=label_home, market_title=g["title"],
                neg_risk=g["neg_risk"],
            )

        # Check for orphaned positions from a previous run
        if not self.dry_run:
            await self._restore_positions()

        print(f"Tracking {len(self.watchers)} tokens across {len(games)} games:\n")
        for g in games:
            ts = g.get("game_start") or 0
            t = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M UTC") if ts else "no time"
            print(f"  [{t}]  {g['title'][:70]}")
        print()

        scan_count = 0
        while self._running:
            try:
                for token_id, w in list(self.watchers.items()):
                    if w.state == TokenState.CLOSED:
                        continue

                    price = await self._fetch_token_price(token_id)
                    if price is None:
                        continue

                    signal = w.update_price(price, self.config)
                    if signal is None:
                        continue

                    if signal == "ENTRY":
                        await self._handle_entry(w)
                    elif signal == "GAME_OVER" and w.state == TokenState.WATCHING:
                        w.state = TokenState.CLOSED
                        logger.info("[%s] Game over before entry", w.label)
                    elif signal == "GAME_OVER" and w.state == TokenState.IN_POSITION:
                        # No sell order — Polymarket settles automatically
                        if price >= cfg.game_over_high:
                            pnl = (1.0 - w.entry_price) * w.shares
                        else:
                            pnl = (0.0 - w.entry_price) * w.shares
                        w.pnl = pnl
                        pnl_pct = (pnl / (w.entry_price * w.shares)) * 100 if w.entry_price > 0 else 0
                        print(f"  [GAME OVER] {w.label} | settlement PnL=${pnl:+.2f} ({pnl_pct:+.1f}%)")
                        self._log_trade(w, "GAME_OVER", w.shares, w.current_price, pnl, pnl_pct)
                        w.state = TokenState.CLOSED
                    elif signal in ("EXIT_TP", "EXIT_SL"):
                        await self._handle_exit(w, signal)
                        # Re-entry: reset to WATCHING if game still active
                        if 0.01 < price < 0.99:
                            self._reset_watcher(w)

                scan_count += 1
                # Heartbeat every 60s (12 scans × 5s)
                if scan_count % 12 == 0:
                    active = sum(1 for w in self.watchers.values() if w.state == TokenState.WATCHING)
                    in_pos = sum(1 for w in self.watchers.values() if w.state == TokenState.IN_POSITION)
                    closed = sum(1 for w in self.watchers.values() if w.state == TokenState.CLOSED)
                    best = max(
                        (w for w in self.watchers.values() if w.state == TokenState.WATCHING and w.current_price > 0),
                        key=lambda w: w.current_price, default=None,
                    )
                    best_info = f"  best={best.label}:{best.current_price:.4f}" if best else ""
                    print(f"  [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] scan #{scan_count} | "
                          f"watching={active} holding={in_pos} closed={closed}{best_info}")

                await asyncio.sleep(cfg.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Main loop error: %s", e)
                await asyncio.sleep(cfg.poll_interval)

        self._print_summary()

        if self._owns_gamma:
            try:
                await self.gamma.close()
            except Exception:
                pass

    def _print_summary(self) -> None:
        """Print trade summary and export path.

        Includes both realized PnL (closed positions, including auto-settled
        game-over) and unrealized PnL (open positions, mark-to-market).
        """
        closed = [w for w in self.watchers.values() if w.state == TokenState.CLOSED and w.pnl != 0]
        open_pos = [w for w in self.watchers.values() if w.state == TokenState.IN_POSITION]
        realized_pnl = sum(w.pnl for w in closed)

        # Unrealized MTM for open positions
        unrealized_pnl = 0.0
        for w in open_pos:
            if w.entry_price > 0:
                unrealized_pnl += (w.current_price - w.entry_price) * w.shares

        total_pnl = realized_pnl + unrealized_pnl

        print(f"\n{'='*60}")
        print("TRADE SUMMARY")
        print(f"{'='*60}")
        print(f"  Trades closed:        {len(closed)} (realized PnL: ${realized_pnl:+.2f})")
        print(f"  Positions open:       {len(open_pos)} (unrealized MTM: ${unrealized_pnl:+.2f})")
        print(f"  Total PnL:           ${total_pnl:+.2f}")
        if closed:
            wins = sum(1 for w in closed if w.pnl > 0)
            losses = sum(1 for w in closed if w.pnl < 0)
            settled = sum(1 for w in closed if w.order_id == "" and w.pnl != 0)
            print(f"  Wins / Losses:        {wins} / {losses}  (auto-settled: {settled})")
        if open_pos:
            print()
            for w in open_pos:
                if w.entry_price > 0:
                    mtm = (w.current_price - w.entry_price) * w.shares
                    mtm_pct = (w.current_price - w.entry_price) / w.entry_price * 100
                    print(f"  [{w.label}] {w.side.upper()} entry={w.entry_price:.4f} "
                          f"now={w.current_price:.4f} | MTM=${mtm:+.2f} ({mtm_pct:+.1f}%)")
        print(f"  Trade log:            {self._trade_log_path.resolve()}")
        if self._db:
            print(f"  Database:             trading_system.db")
        print(f"{'='*60}\n")

    async def _restore_positions(self) -> None:
        """Restore open positions from a previous bot run.

        Queries the Polymarket Data API for open positions of the funder.
        For any active position whose condition_id matches a discovered
        game, creates a TokenWatcher in IN_POSITION state so the bot
        resumes monitoring (take-profit / stop-loss / game-over).
        """
        try:
            resp = await self.client.get_positions()
        except Exception as e:
            logger.warning("Could not fetch positions: %s", e)
            return

        positions = resp.get("market_positions", [])
        active = [p for p in positions if float(p.get("size", 0)) > 0]
        if not active:
            return

        print(f"\nFound {len(active)} open position(s) from previous run:")
        for p in active:
            cond = p.get("condition_id", "")
            token_id = p.get("token_id", "")
            side = p.get("side", "").lower()
            size = int(float(p.get("size", 0)))
            avg_price = float(p.get("avg_price", 0))

            # Only restore if this is in our watchers (MLB game)
            w = self.watchers.get(token_id)
            if w is None:
                self.watchers[token_id] = TokenWatcher(
                    token_id=token_id, condition_id=cond,
                    side=side, label=f"{cond[:10]}... ({side.upper()})",
                    market_title=cond, neg_risk=False,
                    state=TokenState.IN_POSITION, entry_price=avg_price,
                    peak_price=avg_price, shares=size,
                )
                w = self.watchers[token_id]

            w.state = TokenState.IN_POSITION
            w.entry_price = avg_price
            w.peak_price = avg_price
            w.shares = size
            print(f"  [{w.label}] {side.upper()} entry={avg_price:.4f} x{size}")

        print()

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _reset_watcher(self, w: TokenWatcher) -> None:
        """Reset a closed watcher so it can re-enter on the same game."""
        w.state = TokenState.WATCHING
        w.entry_price = 0.0
        w.peak_price = 0.0
        w.shares = 0
        w.order_id = ""
        w.pnl = 0.0
        w.cooldown_scans = 5  # ~15s at 3s poll — avoid instant re-entry
        logger.info("[%s] Reset to WATCHING — can re-enter after cooldown", w.label)

    def _compute_shares(self) -> int:
        return max(1, int(self.config.trade_size))

    def _log_trade(
        self, w: TokenWatcher, action: str, shares: int, price: float,
        pnl: float, pnl_pct: float, ts: datetime | None = None,
    ) -> None:
        """Write trade event to both the text log file and the database."""
        ts = ts or datetime.now(timezone.utc)
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        line = (
            f"[{ts_str}] {action:<8} | {w.label:<30} | {w.side.upper():>4} "
            f"@ {price:.4f} x {shares} | PnL=${pnl:+.2f} ({pnl_pct:+.1%})"
            f" | {w.market_title[:60]}"
        )
        try:
            with open(self._trade_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

        # Write to database (only when position closes or entry)
        if self._db is not None:
            try:
                if action in ("EXIT_TP", "EXIT_SL", "EXIT_GAMEOVER"):
                    entry_ts = self._open_trades.pop(w.token_id, ts)
                    trade = TradeLog(
                        market_id=w.condition_id,
                        side=w.side,
                        entry_price=w.entry_price,
                        exit_price=price,
                        quantity=shares,
                        pnl=pnl,
                        entry_timestamp=entry_ts,
                        exit_timestamp=ts,
                        rationale=action,
                        strategy="late_game_favorite",
                    )
                    asyncio.ensure_future(self._db.add_trade_log(trade))
            except Exception:
                pass

    async def _handle_entry(self, w: TokenWatcher) -> None:
        cfg = self.config
        if w.current_price <= 0:
            return
        shares = self._compute_shares()
        entry_price = w.current_price
        entry_time = datetime.now(timezone.utc)

        logger.info("[%s] ENTRY — price=%.4f shares=%d cost=$%.2f",
                     w.label, entry_price, shares, shares * entry_price)

        if self.dry_run:
            print(f"  [DRY] BUY {shares} {w.side.upper()} [{w.label}] "
                  f"@ {entry_price:.4f} (~${shares * entry_price:.2f})")
            w.state = TokenState.IN_POSITION
            w.entry_price = entry_price
            w.peak_price = entry_price
            w.shares = shares
            self._open_trades[w.token_id] = entry_time
            self._log_trade(w, "ENTRY", shares, entry_price, 0, 0, entry_time)
            return

        try:
            price_cents = int(round(entry_price * 100))
            params: Dict[str, Any] = {
                "ticker": w.condition_id,
                "client_order_id": str(uuid.uuid4()),
                "side": w.side,
                "action": "buy",
                "count": shares,
                "type_": "market",
            }
            if w.side == "yes":
                params["yes_price"] = price_cents
            else:
                params["no_price"] = price_cents

            resp = await self.client.place_order(**params)
            order_id = resp.get("order", {}).get("order_id", "")

            w.state = TokenState.IN_POSITION
            w.entry_price = entry_price
            w.shares = shares
            w.order_id = order_id
            print(f"  LIVE BUY {shares} {w.side.upper()} [{w.label}] "
                  f"@ {entry_price:.4f} | {order_id}")
            self._open_trades[w.token_id] = entry_time
            self._log_trade(w, "ENTRY", shares, entry_price, 0, 0, entry_time)
        except Exception as e:
            logger.error("[%s] Entry failed: %s | price=%.4f shares=%d side=%s",
                         w.label, e, w.current_price, shares, w.side)

    async def _get_actual_shares(self, token_id: str) -> float:
        """Query Polymarket Data API for the actual share count held."""
        try:
            resp = await self.client.get_positions()
            for p in resp.get("market_positions", []):
                if p.get("token_id") == token_id:
                    return float(p.get("size", 0))
        except Exception:
            pass
        return 0.0

    async def _handle_exit(self, w: TokenWatcher, signal: str) -> None:
        exit_price = w.current_price
        exit_time = datetime.now(timezone.utc)

        # Sell the actual position size, not the expected count
        actual_shares = float(w.shares)
        if not self.dry_run:
            pos_count = await self._get_actual_shares(w.token_id)
            if pos_count > 0:
                actual_shares = pos_count
        sell_count = actual_shares
        if sell_count <= 0:
            logger.warning("[%s] No shares to sell, marking closed", w.label)
            w.state = TokenState.CLOSED
            return

        if w.entry_price > 0:
            pnl = (exit_price - w.entry_price) * actual_shares
            pnl_pct = (exit_price - w.entry_price) / w.entry_price * 100
        else:
            pnl, pnl_pct = 0.0, 0.0

        reasons = {
            "EXIT_TP": f"Take-profit +{pnl_pct:.0f}%",
            "EXIT_SL": f"Stop-loss {pnl_pct:.0f}%",
            "EXIT_GAMEOVER": "Game over",
        }
        reason = reasons.get(signal, signal)

        logger.info("[%s] %s — price=%.4f shares=%.3f sell=%d pnl=$%.2f",
                    w.label, reason, exit_price, actual_shares, sell_count, pnl)

        if self.dry_run:
            print(f"  [DRY] {reason}: SELL {sell_count} {w.side.upper()} [{w.label}] "
                  f"@ {exit_price:.4f} | PnL=${pnl:+.2f}")
            w.state = TokenState.CLOSED
            w.pnl = pnl
            self._log_trade(w, signal, actual_shares, exit_price, pnl, pnl_pct, exit_time)
            return

        try:
            price_cents = int(round(exit_price * 100))
            await self.client.place_order(
                ticker=w.condition_id,
                client_order_id=str(uuid.uuid4()),
                side=w.side,
                action="sell",
                count=sell_count,
                type_="limit",
                yes_price=price_cents if w.side == "yes" else None,
                no_price=price_cents if w.side == "no" else None,
            )
            print(f"  LIVE {reason}: SELL {w.shares} {w.side.upper()} [{w.label}] "
                  f"@ {exit_price:.4f} | PnL=${pnl:+.2f}")
            w.state = TokenState.CLOSED
            w.pnl = pnl
        except Exception as e:
            logger.error("[%s] Exit failed: %s", w.label, e)
