#!/usr/bin/env python3
"""Quick smoke test for MLB late-game strategy — Gamma discovery + price polling.

Does NOT import PolymarketClient or py-clob-client. Uses httpx directly
to validate the two critical data paths: market discovery and price polling.
"""

import asyncio
import json
import sys
import time as _time
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

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


def is_mlb_moneyline(title: str) -> bool:
    t = title.lower()
    if any(kw in t for kw in SKIP_TITLE_KEYWORDS):
        return False
    if not any(kw in t for kw in MLB_TEAM_KEYWORDS):
        return False
    return any(w in t for w in [" vs ", " vs. ", " v ", " @ ", " at ",
                                "to beat", "beats ", "will beat", "will win"])


async def fetch_midpoint(token_id: str) -> float | None:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as sess:
            r = await sess.get(
                f"{CLOB}/midpoint",
                params={"token_id": token_id, "_": int(_time.time() * 1000)},
            )
            if r.status_code == 200:
                v = r.json().get("mid")
                return float(v) if v is not None else None
    except Exception as e:
        print(f"  Price fetch error for {token_id[:8]}...: {e}")
    return None


async def discover_mlb_markets() -> list[dict]:
    """Query Gamma /events with tag_id=6 (baseball), filter to MLB moneyline."""
    print("=" * 60)
    print("STEP 1: Discover MLB moneyline markets from Gamma API")
    print("=" * 60)

    markets = []
    seen = set()

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as sess:
        for page in range(5):
            params = {
                "active": "true",
                "closed": "false",
                "limit": 200,
                "offset": page * 200,
                "order": "volume",
                "ascending": "false",
                "tag_id": 6,  # baseball
            }
            try:
                r = await sess.get(f"{GAMMA}/events", params=params)
                if r.status_code != 200:
                    break
                data = r.json()
                if not isinstance(data, list) or not data:
                    break

                for event in data:
                    for m in event.get("markets", []) or []:
                        cond = m.get("conditionId") or ""
                        title = m.get("question") or ""

                        if not cond or cond in seen:
                            continue
                        if not is_mlb_moneyline(title):
                            continue
                        if not m.get("acceptingOrders", True):
                            continue

                        seen.add(cond)

                        # Parse token IDs
                        raw_ids = m.get("clobTokenIds") or "[]"
                        try:
                            ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
                            yes_id, no_id = (str(ids[0]), str(ids[1])) if len(ids) >= 2 else ("", "")
                        except (json.JSONDecodeError, TypeError):
                            yes_id, no_id = "", ""

                        vol = float(m.get("volumeNum") or m.get("volume") or 0)
                        markets.append({
                            "condition_id": cond,
                            "title": title,
                            "yes_token": yes_id,
                            "no_token": no_id,
                            "volume": vol,
                        })
            except Exception as e:
                print(f"  Gamma page {page} error: {e}")
                break

    markets.sort(key=lambda x: -x["volume"])
    print(f"  Found {len(markets)} MLB moneyline markets\n")
    return markets


async def poll_prices(markets: list[dict]) -> None:
    """Poll midpoint prices for all YES/NO tokens and simulate strategy signals."""
    print("=" * 60)
    print("STEP 2: Poll token prices and simulate signals")
    print("=" * 60)

    # Collect all unique token IDs
    token_map = {}  # token_id → (label, market_title)
    for m in markets[:5]:  # top 5 by volume
        import re
        title = m["title"]
        # extract teams
        pats = [
            r"will\s+(.+?)\s+(?:win\s+(?:vs?\.?\s+|against\s+|over\s+)|beat\s+|defeat\s+)(.+?)(?:\?|$|\s+on|\s+\()",
            r"^(.+?)\s+vs\.?\s+(.+?)(?:\?|$|\s+on|\s+[-–\(])",
            r"^(.+?)\s+v\s+(.+?)(?:\?|$|\s+on|\s+[-–\(])",
            r"^(.+?)\s+(?:at|@)\s+(.+?)(?:\?|$|\s+on|\s+[-–\(])",
        ]
        away, home = "Team A", "Team B"
        for pat in pats:
            m2 = re.search(pat, title, re.I)
            if m2:
                away, home = m2.group(1).strip()[:30], m2.group(2).strip()[:30]
                break

        if m["yes_token"]:
            token_map[m["yes_token"]] = (f"{away} (AWAY YES)", title)
        if m["no_token"]:
            token_map[m["no_token"]] = (f"{home} (HOME NO)", title)

    print(f"  Polling {len(token_map)} tokens...\n")

    for tid, (label, title) in token_map.items():
        price = await fetch_midpoint(tid)
        icon = ""
        if price is not None:
            if price >= 0.99:
                icon = "GAME OVER (win)"
            elif price <= 0.01:
                icon = "GAME OVER (loss)"
            elif price >= 0.80:
                icon = "ENTRY SIGNAL"
            pnl_info = ""
            if 0.01 < price < 0.99 and price >= 0.80:
                pnl_info = f" | TP target: {price * 1.8:.4f}  SL target: {price * 0.87:.4f}"
            print(f"  [{label:<30}] price={price:.4f}  {icon}{pnl_info}")
            print(f"    {title[:80]}")
        else:
            print(f"  [{label:<30}] price=N/A (fetch failed)")
            print(f"    {title[:80]}")
        await asyncio.sleep(0.15)


async def main():
    markets = await discover_mlb_markets()
    if not markets:
        print("No MLB moneyline markets found. Possible reasons:")
        print("  1. No MLB games today")
        print("  2. Gamma tag_id=6 doesn't match baseball")
        print("  3. Market titles don't match our keywords")
        print("\nTop 10 market titles from Gamma (any tag):")
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as sess:
            r = await sess.get(f"{GAMMA}/events", params={
                "active": "true", "closed": "false", "limit": 10,
                "order": "volume", "ascending": "false",
            })
            if r.status_code == 200:
                for evt in r.json()[:5]:
                    for m in (evt.get("markets", []) or [])[:2]:
                        tags = [t.get("slug","") for t in (evt.get("tags") or [])]
                        print(f"  [{','.join(tags[:3])}] {m.get('question','')[:80]}")
        return

    # Show summary
    print(f"\n{'─'*60}")
    print("TOP MARKETS BY VOLUME:")
    for m in markets[:10]:
        print(f"  ${m['volume']:>8.0f}  {m['title'][:70]}")
    print()

    await poll_prices(markets)


if __name__ == "__main__":
    asyncio.run(main())
