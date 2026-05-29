#!/usr/bin/env python3
"""
One-time USDC + CTF allowance provisioner for the Polymarket trading wallet.

Polymarket's CLOB matches orders against three on-chain contracts:
  - CTF Exchange       (legacy)
  - Neg-Risk Exchange  (newer markets)
  - Neg-Risk Adapter   (used by both)

For each, the funding wallet must:
  1. `approve(spender, MAX)` on USDC.e (so the contract can pull funds)
  2. `setApprovalForAll(spender, true)` on the Conditional-Tokens ERC1155
     (so the contract can move your YES/NO shares when you sell)

This script reads your wallet from .env (POLYMARKET_PRIVATE_KEY / FUNDER /
SIGNATURE_TYPE / POLYGON_RPC_URL), checks current allowance state, and only
sends transactions for what's missing. Idempotent — safe to re-run.

Usage:
    python scripts/set_allowances.py            # dry-run preview
    python scripts/set_allowances.py --send     # actually broadcast txs

Reference (canonical example):
    https://gist.github.com/poly-rodr/44313920481de58d5a3f6d1f8226bd5e
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `src` importable when invoked as `python scripts/set_allowances.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.clients.polymarket_client import (  # noqa: E402
    POLYMARKET_SPENDERS,
    USDC_E_POLYGON,
    DEFAULT_RPC_URL,
    ALLOWANCE_OK_THRESHOLD,
)


# Conditional Tokens Framework (ERC1155) on Polygon — Polymarket's outcome-token
# contract. Source: py-clob-client constants + Polymarket docs.
CTF_ERC1155_POLYGON = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# unlimited approval value (uint256 max)
MAX_UINT256 = 2**256 - 1

ERC20_APPROVE_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
]

ERC1155_APPROVAL_ABI = [
    {
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"},
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "operator", "type": "address"},
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def _build_w3(rpc_url: str):
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise SystemExit(f"Cannot reach Polygon RPC at {rpc_url}")
    return w3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--send",
        action="store_true",
        help="Actually broadcast approval transactions (default: dry-run)",
    )
    args = parser.parse_args()

    load_dotenv()

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
    funder_env = os.getenv("POLYMARKET_FUNDER", "").strip()
    signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))
    rpc_url = os.getenv("POLYGON_RPC_URL", "") or DEFAULT_RPC_URL

    if not private_key:
        print("ERROR: POLYMARKET_PRIVATE_KEY is not set in .env")
        return 1

    if signature_type != 0:
        print(
            "INFO: SIGNATURE_TYPE != 0 (proxy wallet). Magic-Link / Safe wallets\n"
            "      generally have allowances set automatically by Polymarket. If\n"
            "      orders still fail with 'allowance' errors, run this with --send."
        )

    from eth_account import Account
    acct = Account.from_key(private_key)
    signer = acct.address
    funder = funder_env or signer

    print("=" * 64)
    print("  POLYMARKET ALLOWANCE PROVISIONING")
    print("=" * 64)
    print(f"  RPC:             {rpc_url}")
    print(f"  Signer address:  {signer}")
    print(f"  Funder address:  {funder}")
    print(f"  Mode:            {'BROADCAST' if args.send else 'DRY RUN'}")
    print()

    if funder != signer and signature_type == 0:
        print(
            "WARN: SIGNATURE_TYPE=0 (EOA) but POLYMARKET_FUNDER differs from\n"
            "      the signer address. Approvals are set FROM the signer's wallet,\n"
            "      not the funder's. Re-check your .env if this is wrong."
        )
        print()

    w3 = _build_w3(rpc_url)

    usdc = w3.eth.contract(
        address=w3.to_checksum_address(USDC_E_POLYGON),
        abi=ERC20_APPROVE_ABI,
    )
    ctf = w3.eth.contract(
        address=w3.to_checksum_address(CTF_ERC1155_POLYGON),
        abi=ERC1155_APPROVAL_ABI,
    )

    pending_txs = []

    # -----------------------------------------------------------------
    # USDC.e approvals
    # -----------------------------------------------------------------
    print("USDC.e approvals:")
    for label, spender in POLYMARKET_SPENDERS.items():
        try:
            current = usdc.functions.allowance(
                w3.to_checksum_address(signer),
                w3.to_checksum_address(spender),
            ).call()
        except Exception as exc:
            print(f"  [{label}] read failed: {exc}")
            continue

        if current >= ALLOWANCE_OK_THRESHOLD:
            print(f"  [{label}] OK ({current / 10**6:,.0f} USDC) — no action")
            continue

        print(f"  [{label}] needs approval (current: {current / 10**6:,.0f} USDC)")
        pending_txs.append(
            ("usdc.approve", label, spender,
             usdc.functions.approve(w3.to_checksum_address(spender), MAX_UINT256))
        )

    # -----------------------------------------------------------------
    # CTF (ERC1155) operator approvals
    # -----------------------------------------------------------------
    print()
    print("Conditional Tokens (ERC1155) operator approvals:")
    for label, spender in POLYMARKET_SPENDERS.items():
        try:
            approved = ctf.functions.isApprovedForAll(
                w3.to_checksum_address(signer),
                w3.to_checksum_address(spender),
            ).call()
        except Exception as exc:
            print(f"  [{label}] read failed: {exc}")
            continue

        if approved:
            print(f"  [{label}] OK — already operator-approved")
            continue

        print(f"  [{label}] needs setApprovalForAll(true)")
        pending_txs.append(
            ("ctf.setApprovalForAll", label, spender,
             ctf.functions.setApprovalForAll(w3.to_checksum_address(spender), True))
        )

    if not pending_txs:
        print()
        print("All allowances already set. Nothing to do.")
        return 0

    print()
    print(f"  {len(pending_txs)} transaction(s) needed.")
    if not args.send:
        print("  Re-run with --send to broadcast them.")
        return 0

    # -----------------------------------------------------------------
    # Broadcast
    # -----------------------------------------------------------------
    chain_id = w3.eth.chain_id
    nonce = w3.eth.get_transaction_count(w3.to_checksum_address(signer))

    for kind, label, spender, fn in pending_txs:
        gas_price = w3.eth.gas_price
        try:
            tx = fn.build_transaction({
                "from": w3.to_checksum_address(signer),
                "nonce": nonce,
                "gas": 200_000,
                "gasPrice": gas_price,
                "chainId": chain_id,
            })
            signed = acct.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  {kind} [{label}] → {tx_hash.hex()}  (nonce {nonce}, gas {gas_price})")
            nonce += 1
        except Exception as exc:
            print(f"  ERROR: {kind} [{label}] failed: {exc}")
            return 2

    print()
    print("Done. Wait ~30s for confirmations, then run:")
    print("    python -c 'import asyncio; from src.clients.polymarket_client import PolymarketClient; "
          "print(asyncio.run(PolymarketClient().check_allowances()))'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
