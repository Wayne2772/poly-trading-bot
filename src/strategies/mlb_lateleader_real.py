"""
MLB Late-Leader Real Strategy — delayed entry, absolute take-profit, fixed stop-loss.

Strategy:
  - Monitor both sides (YES/NO tokens) per MLB moneyline game independently.
  - Entry only after `entry_delay_minutes` past game start.
  - Enter when token price > entry_confidence (0.55).
  - Exit on: absolute take-profit price (0.93), fixed stop-loss (15% below entry),
    or game-over (>=0.995 or <=0.005, no sell — Polymarket settles).

Differences from late_game_favorite:
  - entry_delay_minutes instead of immediate entry after discovery
  - Absolute take-profit price (0.93) instead of percentage from entry
  - Fixed stop-loss from entry_confidence baseline instead of entry price
  - No peak_price tracking (no trailing stop)
"""

from __future__ import annotations

import asyncio
import json
import re
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from src.clients.gamma_client import GammaClient
from src.clients.polymarket_client import PolymarketClient
from src.utils.database import DatabaseManager, TradeLog
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("lateleader_real")

# ---------------------------------------------------------------------------
# Market filtering (shared with late_game_favorite)
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
class MLBLateLeaderRealConfig:
    entry_delay_minutes: int = 60     # minutes after game start before entry allowed
    entry_confidence: float = 0.56    # price must be > this to enter
    take_profit_price: float = 0.93   # absolute price — exit when price >= this
    stop_loss_pct: float = 0.15       # fixed % below entry: exit when price <= entry * (1 - this)
    game_over_high: float = 0.995     # game decided (win)
    game_over_low: float = 0.005      # game decided (loss)
    trade_size: float = 6             # shares per trade
    poll_interval: int = 1            # seconds between price polls
    max_games: int = 5               # max concurrent games to track
    max_concurrent_polls: int = 10    # 并发拉取 parallel token price requests per scan


# ---------------------------------------------------------------------------
# TokenWatcher — per-token state machine
# ---------------------------------------------------------------------------

class TokenState(Enum):
    WATCHING = "watching"
    IN_POSITION = "in_position"
    CLOSED = "closed"


@dataclass
class TokenWatcher:
    token_id: str
    condition_id: str
    side: str               # "yes" or "no"
    label: str              # e.g. "Dodgers (06-02 01:40)"
    market_title: str
    game_start_time: float = 0.0  # unix timestamp, 0 = unknown
    neg_risk: bool = False
    state: TokenState = TokenState.WATCHING
    current_price: float = 0.0
    entry_price: float = 0.0
    shares: int = 0
    order_id: str = ""
    pnl: float = 0.0
    pending_action: str = ""  # signal awaiting confirmation
    pending_countdown: int = 0  # remaining confirmations needed (2 = two more scans)
    _skip_confirm: bool = field(default=False, repr=False)  # skip confirmation after FOK failure
    _delay_warned: bool = field(default=False, repr=False)

    def _confirm_next_scan(self, signal: str) -> Optional[str]:
        """Return `signal` after 1 confirmation (2 scans total), or immediately if _skip_confirm."""
        if self._skip_confirm:
            self._skip_confirm = False
            self.pending_action = ""; self.pending_countdown = 0; self._skip_confirm = False
            return signal
        if self.pending_action == signal and self.pending_countdown > 0:
            self.pending_countdown -= 1
            if self.pending_countdown == 0:
                self.pending_action = ""; self.pending_countdown = 0; self._skip_confirm = False
                return signal
            return None
        self.pending_action = signal
        self.pending_countdown = 1  # need 1 more confirmation (2 scans total)
        return None

    def update_price(
        self, price: float, cfg: "MLBLateLeaderRealConfig", now_ts: float,
    ) -> Optional[str]:
        """Return signal string or None. `now_ts` is current unix time."""
        self.current_price = price

        if self.state == TokenState.WATCHING:
            # Game-over check first (immediate — definitive)
            if price <= cfg.game_over_low or price >= cfg.game_over_high:
                self.pending_action = ""; self.pending_countdown = 0; self._skip_confirm = False
                return "GAME_OVER"

            # Entry delay: must be past game_start + entry_delay_minutes
            if self.game_start_time > 0:
                eligible_at = self.game_start_time + cfg.entry_delay_minutes * 60
                if now_ts < eligible_at:
                    if not self._delay_warned:
                        remaining = int((eligible_at - now_ts) / 60) + 1
                        logger.info(
                            "[%s] Waiting — entry eligible in ~%d min (after %s UTC)",
                            self.label, remaining,
                            datetime.fromtimestamp(eligible_at, tz=timezone.utc).strftime("%H:%M"),
                        )
                        self._delay_warned = True
                    return None
            else:
                # No game start time — skip this token permanently
                if not self._delay_warned:
                    logger.warning(
                        "[%s] SKIPPED — no gameStartTime from Gamma, cannot determine entry window",
                        self.label,
                    )
                    self._delay_warned = True
                    self.state = TokenState.CLOSED
                return None

            # Past delay window — entry must have room to TP
            if cfg.entry_confidence < price < cfg.take_profit_price:
                return self._confirm_next_scan("ENTRY")
            self.pending_action = ""; self.pending_countdown = 0; self._skip_confirm = False

        elif self.state == TokenState.IN_POSITION:
            if price <= cfg.game_over_low or price >= cfg.game_over_high:
                self.pending_action = ""; self.pending_countdown = 0; self._skip_confirm = False
                return "GAME_OVER"

            if self.entry_price > 0:
                # Absolute take-profit
                if price >= cfg.take_profit_price:
                    return self._confirm_next_scan("EXIT_TP")
                # Fixed stop-loss from entry-confidence baseline
                stop_price = cfg.entry_confidence * (1.0 - cfg.stop_loss_pct)
                if price <= stop_price:
                    return self._confirm_next_scan("EXIT_SL")
            self.pending_action = ""; self.pending_countdown = 0; self._skip_confirm = False

        return None


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MLBLateLeaderRealStrategy:
    """MLB late-leader real — delayed entry, absolute TP, fixed SL."""

    def __init__(
        self,
        client: PolymarketClient,
        gamma: Optional[GammaClient] = None,
        config: Optional[MLBLateLeaderRealConfig] = None,
        dry_run: bool = True,
        db: Optional[DatabaseManager] = None,
    ):
        self.client = client
        self.gamma = gamma or GammaClient()
        self._owns_gamma = gamma is None
        self.config = config or MLBLateLeaderRealConfig()
        self.dry_run = dry_run
        self.watchers: Dict[str, TokenWatcher] = {}
        self._running = False
        self._db = db
        self._trade_log_path = Path(__file__).resolve().parent.parent.parent / "mlb_lateleader_real_trades.log"
        self._open_trades: Dict[str, datetime] = {}
        self._blocklist_path = Path(__file__).resolve().parent.parent.parent / "mlb_lateleader_real_blocklist.txt"
        self._blocked: set[str] = set()
        self._blocklist_mtime: float = 0.0
        self._output_dir = Path(__file__).resolve().parent.parent.parent / "output"
        self._price_files: Dict[str, Path] = {}  # condition_id → jsonl path
        self._realized_pnl: float = 0.0
        self._total_wins: int = 0
        self._total_losses: int = 0

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

        # Sort: in-progress first, then upcoming, then ended, then no-time
        now = datetime.now(timezone.utc).timestamp()
        GAME_MAX_SECONDS = 5 * 3600  # games >5h old are considered ended

        def _sort_key(m: dict) -> tuple:
            ts = m.get("game_start") or 0
            if ts == 0:
                return (3, 0)                   # last: no start time
            if ts <= now:
                if now - ts < GAME_MAX_SECONDS:
                    return (0, now - ts)         # 1st: in-progress, recent first
                return (2, now - ts)             # 3rd: ended
            return (1, ts - now)                 # 2nd: upcoming, soonest first

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

    async def _fetch_prices_batch(
        self, tokens: list[tuple[str, TokenWatcher]],
    ) -> list[tuple[str, TokenWatcher, Optional[float]]]:
        """Fetch prices for multiple tokens in parallel with concurrency limit."""
        sem = asyncio.Semaphore(self.config.max_concurrent_polls)

        async def _one(token_id: str, w: TokenWatcher) -> tuple[str, TokenWatcher, Optional[float]]:
            async with sem:
                price = await self._fetch_token_price(token_id)
            return token_id, w, price

        return await asyncio.gather(*[_one(tid, w) for tid, w in tokens])

    def _write_price(self, token_id: str, w: TokenWatcher, price: float) -> None:
        """Append one price observation to the game's JSONL file."""
        filepath = self._price_files.get(w.condition_id)
        if not filepath:
            return
        record = json.dumps({
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "token_id": token_id,
            "label": w.label,
            "side": w.side,
            "price": price,
        })
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(record + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        cfg = self.config
        mode = "DRY RUN" if self.dry_run else "LIVE"

        print(f"\n{'='*60}")
        print(f"MLB LATE-LEADER REAL — {mode}")
        print(f"  Entry delay:  {cfg.entry_delay_minutes} min after game start")
        print(f"  Entry price:  > {cfg.entry_confidence}")
        print(f"  Take-profit:  price >= {cfg.take_profit_price}")
        print(f"  Stop-loss:    -{cfg.stop_loss_pct*100:.0f}% from entry (fixed)")
        print(f"  Game-over:    >= {cfg.game_over_high} / <= {cfg.game_over_low}")
        print(f"  Size:         {cfg.trade_size:.0f} shares | Poll: {cfg.poll_interval}s")
        print(f"{'='*60}\n")

        games = await self.discover_games()
        if not games:
            print("No MLB moneyline markets found.")
            return

        self.watchers.clear()
        now_ts = datetime.now(timezone.utc).timestamp()
        skipped_no_time = 0

        for g in games:
            away, home = self._extract_teams(g["title"])
            game_ts = g.get("game_start") or 0
            time_str = datetime.fromtimestamp(game_ts, tz=timezone.utc).strftime("%m-%d %H:%M") if game_ts else "no time"
            label_away = f"{away} ({time_str})"
            label_home = f"{home} ({time_str})"

            if game_ts == 0:
                skipped_no_time += 1

            # YES token = bet on away team (first mentioned in "Will X beat Y?")
            self.watchers[g["yes_token"]] = TokenWatcher(
                token_id=g["yes_token"], condition_id=g["condition_id"],
                side="yes", label=label_away, market_title=g["title"],
                game_start_time=game_ts, neg_risk=g["neg_risk"],
            )
            # NO token = bet on home team (second mentioned)
            self.watchers[g["no_token"]] = TokenWatcher(
                token_id=g["no_token"], condition_id=g["condition_id"],
                side="no", label=label_home, market_title=g["title"],
                game_start_time=game_ts, neg_risk=g["neg_risk"],
            )

        if skipped_no_time > 0:
            print(f"⚠️  {skipped_no_time} game(s) have no gameStartTime — these will be skipped at entry.\n")

        # Set up price data output — one JSONL per game
        self._output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        for g in games:
            if g.get("game_start", 0) == 0:
                continue  # no output for games without start time
            cond = g["condition_id"]
            away, home = self._extract_teams(g["title"])
            safe = f"{away}_vs_{home}".replace(" ", "_").replace(".", "").replace("/", "_")[:60]
            filename = f"mlb_prices_{date_str}_{safe}_{cond[:8]}.jsonl"
            self._price_files[cond] = self._output_dir / filename
        print(f"Price data → {self._output_dir}/ ({len(self._price_files)} game files)\n")

        # Check for orphaned positions from a previous run
        if not self.dry_run:
            await self._restore_positions()

        # Summarize entry windows
        print(f"Tracking {len(self.watchers)} tokens across {len(games)} games:\n")
        for g in games:
            ts = g.get("game_start") or 0
            if ts == 0:
                t = "no time — SKIPPED"
            else:
                t = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M UTC")
                eligible = ts + cfg.entry_delay_minutes * 60
                if now_ts < eligible:
                    remaining = int((eligible - now_ts) / 60) + 1
                    t += f"  (entry in ~{remaining} min)"
                else:
                    t += "  (entry window OPEN)"
            print(f"  [{t}]  {g['title'][:70]}")
        print()

        scan_count = 0
        while self._running:
            try:
                self._reload_blocklist()
                now_ts = datetime.now(timezone.utc).timestamp()

                # Collect active (non-blocked, non-closed) tokens
                active: list[tuple[str, TokenWatcher]] = []
                for token_id, w in self.watchers.items():
                    if w.state == TokenState.CLOSED:
                        continue
                    if self._blocked and (token_id in self._blocked or w.label.lower() in self._blocked):
                        continue
                    active.append((token_id, w))

                # Parallel price fetch (semaphore-limited)
                results = await self._fetch_prices_batch(active)

                # Sequential signal processing (avoids race conditions)
                for token_id, w, price in results:
                    if price is None:
                        continue

                    self._write_price(token_id, w, price)

                    signal = w.update_price(price, self.config, now_ts)
                    if signal is None:
                        continue

                    if signal == "ENTRY":
                        await self._handle_entry(w)
                    elif signal == "GAME_OVER" and w.state == TokenState.WATCHING:
                        w.state = TokenState.CLOSED
                        logger.info("[%s] Game over before entry", w.label)
                    elif signal == "GAME_OVER" and w.state == TokenState.IN_POSITION:
                        if price >= cfg.game_over_high:
                            pnl = (1.0 - w.entry_price) * w.shares
                        else:
                            pnl = (0.0 - w.entry_price) * w.shares
                        w.pnl = pnl
                        self._realized_pnl += pnl
                        if pnl > 0: self._total_wins += 1
                        else: self._total_losses += 1
                        pnl_pct = (pnl / (w.entry_price * w.shares)) * 100 if w.entry_price > 0 else 0
                        print(f"  [GAME OVER] {w.label} | settlement PnL=${pnl:+.2f} ({pnl_pct:+.1f}%)")
                        self._log_trade(w, "GAME_OVER", w.shares, w.current_price, pnl, pnl_pct)
                        w.state = TokenState.CLOSED
                    elif signal in ("EXIT_TP", "EXIT_SL"):
                        await self._handle_exit(w, signal)
                        # Re-entry: only after stop-loss (price has room to grow).
                        # After take-profit, price is near 0.93 — no upside left.
                        if w.state == TokenState.CLOSED and 0.01 < price < 0.99:
                            if signal == "EXIT_SL":
                                self._reset_watcher(w)
                            else:
                                logger.info("[%s] TP reached — not re-entering at high price", w.label)

                scan_count += 1
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
        open_pos = [w for w in self.watchers.values() if w.state == TokenState.IN_POSITION]
        unrealized_pnl = 0.0
        for w in open_pos:
            if w.entry_price > 0:
                unrealized_pnl += (w.current_price - w.entry_price) * w.shares

        total_pnl = self._realized_pnl + unrealized_pnl

        print(f"\n{'='*60}")
        print("TRADE SUMMARY")
        print(f"{'='*60}")
        print(f"  Realized PnL:        ${self._realized_pnl:+.2f}")
        print(f"  Wins / Losses:        {self._total_wins} / {self._total_losses}")
        print(f"  Positions open:       {len(open_pos)} (unrealized MTM: ${unrealized_pnl:+.2f})")
        print(f"  Total PnL:           ${total_pnl:+.2f}")
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
        """Restore open positions from a previous bot run."""
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

            # Match by normalized token_id (Gamma=hex, Data API=decimal)
            dec_tid = self._normalize_token_id(token_id)
            w = self.watchers.get(token_id)
            if w is None:
                for tid, cand in self.watchers.items():
                    if self._normalize_token_id(tid) == dec_tid:
                        w = cand
                        break
            if w is None:
                self.watchers[token_id] = TokenWatcher(
                    token_id=token_id, condition_id=cond,
                    side=side, label=f"{cond[:10]}... ({side.upper()})",
                    market_title=cond, neg_risk=False,
                    state=TokenState.IN_POSITION, entry_price=avg_price,
                    shares=size,
                )
                w = self.watchers[token_id]

            w.state = TokenState.IN_POSITION
            w.entry_price = avg_price
            w.shares = size
            print(f"  [{w.label}] {side.upper()} entry={avg_price:.4f} x{size}")

        print()

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _reload_blocklist(self) -> None:
        try:
            stat = self._blocklist_path.stat()
            if stat.st_mtime <= self._blocklist_mtime:
                return
            self._blocklist_mtime = stat.st_mtime
        except FileNotFoundError:
            self._blocked.clear()
            return

        blocked: set[str] = set()
        try:
            for line in self._blocklist_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                blocked.add(line.lower())
        except Exception:
            return
        self._blocked = blocked

    def _reset_watcher(self, w: TokenWatcher) -> None:
        w.state = TokenState.WATCHING
        w.entry_price = 0.0
        w.shares = 0
        w.order_id = ""
        w.pnl = 0.0
        w.pending_action = ""; w.pending_countdown = 0; w._skip_confirm = False
        w._delay_warned = False  # re-check entry window on re-entry
        logger.info("[%s] Reset to WATCHING — ready for re-entry", w.label)

    @staticmethod
    def _is_fatal_error(msg: str) -> bool:
        """Return True for errors that won't be fixed by retrying."""
        fatal = ["not enough balance", "not enough allowance", "insufficient"]
        m = msg.lower()
        return any(kw in m for kw in fatal)

    def _compute_shares(self) -> int:
        return max(1, int(self.config.trade_size))

    def _log_trade(
        self, w: TokenWatcher, action: str, shares: int, price: float,
        pnl: float, pnl_pct: float, ts: datetime | None = None,
    ) -> None:
        ts = ts or datetime.now(timezone.utc)
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        line = (
            f"[{ts_str}] {action:<8} | {w.label:<30} | {w.side.upper():>4} "
            f"@ {price:.4f} x {shares} | PnL=${pnl:+.2f} ({pnl_pct:+.1f}%)"
            f" | {w.market_title[:60]}"
        )
        try:
            with open(self._trade_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

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
                        strategy="mlb_lateleader_real",
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
        sl_price = cfg.entry_confidence * (1.0 - cfg.stop_loss_pct)
        logger.info("[%s] ENTRY — price=%.4f shares=%d cost=$%.2f | SL=%.4f TP=%.4f",
                     w.label, entry_price, shares, shares * entry_price,
                     sl_price, cfg.take_profit_price)

        if self.dry_run:
            print(f"  [DRY] BUY {shares} {w.side.upper()} [{w.label}] "
                  f"@ {entry_price:.4f} (~${shares * entry_price:.2f}) | SL=%.4f TP=%.4f"
                  % (sl_price, cfg.take_profit_price))
            w.state = TokenState.IN_POSITION
            w.entry_price = entry_price
            w.shares = shares
            self._open_trades[w.token_id] = entry_time
            self._log_trade(w, "ENTRY", shares, entry_price, 0, 0, entry_time)
            return

        # Live entry — FOK market order with retry + actual fill verification
        price_cents = int(round(entry_price * 100))
        max_retries = 2
        filled_shares = 0.0
        order_id = ""
        fatal = False
        for attempt in range(max_retries):
            try:
                resp = await self.client.place_order(
                    ticker=w.condition_id,
                    client_order_id=str(uuid.uuid4()),
                    side=w.side,
                    action="buy",
                    count=shares,
                    type_="market",
                    yes_price=price_cents if w.side == "yes" else None,
                    no_price=price_cents if w.side == "no" else None,
                )
                order_id = resp.get("order", {}).get("order_id", "")
            except Exception as e:
                err_msg = str(e)
                logger.error("[%s] Entry FOK rejected (attempt %d/%d): %s",
                           w.label, attempt + 1, max_retries, err_msg)
                if self._is_fatal_error(err_msg):
                    logger.error("[%s] Fatal error — aborting entry", w.label)
                    fatal = True; break
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                continue

            # FOK returned OK — shares are bought (FOK is all-or-nothing)
            filled_shares = shares
            break

        if filled_shares > 0:
            actual = int(filled_shares)
            w.state = TokenState.IN_POSITION
            w.entry_price = entry_price
            w.shares = actual
            w.order_id = order_id
            print(f"  LIVE BUY {actual} {w.side.upper()} [{w.label}] "
                  f"@ {entry_price:.4f} | {order_id}")
            self._open_trades[w.token_id] = entry_time
            self._log_trade(w, "ENTRY", actual, entry_price, 0, 0, entry_time)
        else:
            logger.error("[%s] Entry failed after %d attempts — will retry next scan",
                        w.label, max_retries)
            if not fatal:
                w._skip_confirm = True  # skip confirmation on retry

    @staticmethod
    def _normalize_token_id(token_id: str) -> str:
        tid = token_id.strip()
        if tid.startswith("0x") or tid.startswith("0X"):
            try:
                return str(int(tid, 16))
            except ValueError:
                pass
        return tid

    async def _get_actual_shares(self, token_id: str) -> float:
        dec_tid = self._normalize_token_id(token_id)
        try:
            resp = await self.client.get_positions()
            positions = resp.get("market_positions", [])
            for p in positions:
                ptid = str(p.get("token_id", ""))
                if ptid == dec_tid or ptid == token_id:
                    size = float(p.get("size", 0))
                    logger.info("[_get_actual_shares] found token=%s... size=%.3f", ptid[:20], size)
                    return size
            logger.info("[_get_actual_shares] token=%s... (dec=%s...) NOT found in %d positions",
                       token_id[:20], dec_tid[:20], len(positions))
        except Exception as e:
            logger.warning("[_get_actual_shares] query failed: %s", e)
        return 0.0

    async def _handle_exit(self, w: TokenWatcher, signal: str) -> None:
        exit_price = w.current_price
        exit_time = datetime.now(timezone.utc)

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
            "EXIT_TP": f"Take-profit ${exit_price:.4f}",
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
            self._realized_pnl += pnl
            if pnl > 0: self._total_wins += 1
            else: self._total_losses += 1
            self._log_trade(w, signal, actual_shares, exit_price, pnl, pnl_pct, exit_time)
            return

        # Live exit — FOK market order (FOK is atomic: success = position gone)
        max_retries = 2
        filled = False
        fatal = False
        for attempt in range(max_retries):
            try:
                await self.client.place_order(
                    ticker=w.condition_id,
                    client_order_id=str(uuid.uuid4()),
                    side=w.side,
                    action="sell",
                    count=sell_count,
                    type_="market",
                )
                # FOK returned OK — shares are sold (FOK is all-or-nothing)
                filled = True
                break
            except Exception as e:
                err_msg = str(e)
                logger.error("[%s] FOK rejected (attempt %d/%d): %s",
                           w.label, attempt + 1, max_retries, err_msg)
                if self._is_fatal_error(err_msg):
                    logger.error("[%s] Fatal error — aborting exit", w.label)
                    fatal = True; break
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                continue

        if filled:
            print(f"  LIVE {reason}: SELL {actual_shares:.3f} {w.side.upper()} [{w.label}] "
                  f"@ {exit_price:.4f} | PnL=${pnl:+.2f}")
            w.state = TokenState.CLOSED
            w.pnl = pnl
            self._realized_pnl += pnl
            if pnl > 0: self._total_wins += 1
            else: self._total_losses += 1
            self._log_trade(w, signal, actual_shares, exit_price, pnl, pnl_pct, exit_time)
        else:
            logger.error("[%s] Exit failed after %d attempts — retry next scan",
                        w.label, max_retries)
            if not fatal:
                w._skip_confirm = True  # skip confirmation on retry
            # Keep state IN_POSITION — next scan will detect signal again
