#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
LoCoMo Ingestion Script (plugin-fidelity).

Mirrors how openclaw-plugin's memory_store flow ingests sessions into OpenViking:
    1. createSession
    2. addSessionMessage per turn (multi-turn, alternating user/assistant)
    3. commitSession with client-side task polling (wait=true equivalent)

This script does NOT run QA. It only ingests + reports the resulting memory
inventory. Pair with bench_recall_locomo.py for the recall benchmark.

Usage:
    cd source/
    source .venv/bin/activate

    # Default: sample 0, all sessions, all turns multi-turn, isolated bench tenant
    python tests/integration/bench_ingest_locomo.py

    # Specific sample + session range
    python tests/integration/bench_ingest_locomo.py --sample 0 --sessions 1-5

    # Cleanup leftover bench data first
    python tests/integration/bench_ingest_locomo.py --clean
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure source root is on path
_source_root = str(Path(__file__).resolve().parent.parent.parent)
if _source_root not in sys.path:
    sys.path.insert(0, _source_root)

from openviking_cli.client.http import AsyncHTTPClient as _OVClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERVER_URL = os.environ.get("OPENVIKING_BASE_URL", "http://127.0.0.1:1933")
API_KEY = os.environ.get("OPENVIKING_API_KEY", "openviking-local")
LOCOMO_PATH = os.path.expanduser("~/Documents/Projects/openclaw/openclaw-eval/locomo10.json")
RESULT_DIR = Path(__file__).parent / "bench_results"
OV_CONF_PATH = os.environ.get(
    "OPENVIKING_CONFIG_FILE",
    "/home/nvidia/Documents/Projects/openclaw/openviking/ov.conf",
)

BENCH_ACCOUNT = "bench"
BENCH_USER = "bench-locomo"
BENCH_AGENT = "bench-locomo"

# Per-session commit-poll deadline. The plugin defaults to 120s; LoCoMo sessions
# with 20+ turns can need longer because the LLM extractor walks each turn.
DEFAULT_COMMIT_TIMEOUT_S = 300


# ---------------------------------------------------------------------------
# ov.conf loading — discovers which VLM OpenViking is currently using
# ---------------------------------------------------------------------------
def load_vlm_config() -> Dict[str, Any]:
    """Read ov.conf and return the vlm section.

    Falls back to an empty dict (with a warning) if the file is unreadable.
    The container mounts ov.conf as a bind volume, so reading it from the host
    sees the exact same config the OpenViking server is using.
    """
    try:
        with open(OV_CONF_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"  WARNING: cannot read {OV_CONF_PATH}: {e}", file=sys.stderr)
        return {}
    return cfg.get("vlm", {}) or {}


def print_vlm_banner(vlm: Dict[str, Any]) -> None:
    if not vlm:
        print("  Extraction VLM: (unknown — ov.conf not readable)")
        return
    model = vlm.get("model", "?")
    api_base = vlm.get("api_base", "?")
    provider = vlm.get("provider", "?")
    temperature = vlm.get("temperature", "?")
    thinking = vlm.get("thinking", False)
    print(f"  Extraction VLM: {model}")
    print(f"    api_base:    {api_base}")
    print(f"    provider:    {provider}  temperature: {temperature}  thinking: {thinking}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
def _make_client() -> _OVClient:
    return _OVClient(
        url=SERVER_URL,
        api_key=API_KEY,
        account=BENCH_ACCOUNT,
        user=BENCH_USER,
        agent_id=BENCH_AGENT,
    )


# ---------------------------------------------------------------------------
# LoCoMo loading
# ---------------------------------------------------------------------------
def load_locomo(path: str, sample_index: int) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if sample_index < 0 or sample_index >= len(data):
        raise ValueError(f"sample_index {sample_index} out of range (0-{len(data) - 1})")
    return data[sample_index]


def get_sessions(sample: Dict, session_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
    """Build per-session message lists with speaker → role mapping.

    speaker_a → "user"        (the protagonist Caroline in LoCoMo sample 0)
    speaker_b → "assistant"   (the conversation partner Melanie)
    """
    conv = sample["conversation"]
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]

    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    sessions: List[Dict] = []
    for sk in session_keys:
        num = int(sk.split("_")[1])
        if session_range and (num < session_range[0] or num > session_range[1]):
            continue

        dt_key = f"{sk}_date_time"
        date_time = conv.get(dt_key, "")

        # Plugin-fidelity formatting: mirror how openclaw-plugin's
        # extractNewTurnTexts (text-utils.ts:420) + addSessionMessage
        # (context-engine.ts:842) deliver turns to OpenViking. Production sends
        # every message as role="user" with a "[sender]: text" inline prefix —
        # never role="assistant". For LoCoMo (two-human conversation), use the
        # real speaker names as the inline tag so the extractor's Speaker
        # Attribution rule can keep Caroline's and Melanie's facts apart.
        messages = []
        for msg in conv[sk]:
            messages.append({
                "role": "user",
                "text": f"[{msg['speaker']}]: {msg['text']}",
            })

        sessions.append({
            "session_key": sk,
            "date_time": date_time,
            "messages": messages,
            "speakers": f"{speaker_a} / {speaker_b} (both sent as role=user with [speaker]: prefix)",
        })

    return sessions


# ---------------------------------------------------------------------------
# Ingestion (mirrors plugin's createSession + addSessionMessage + commitSession)
# ---------------------------------------------------------------------------
async def _commit_with_polling(
    client: _OVClient,
    session_id: str,
    timeout_s: int,
) -> Dict[str, Any]:
    """Mirror of plugin's commitSession({ wait: true }).

    The SDK's commit_session returns immediately with a task_id; we poll
    /api/v1/tasks/{task_id} every 500ms until status=completed/failed
    or until timeout. Returns a merged status dict for printing/telemetry.
    """
    commit_res = await client.commit_session(session_id, telemetry=True)
    if not isinstance(commit_res, dict):
        return {"status": "unknown"}

    task_id = commit_res.get("task_id")
    if not task_id:
        # Server already finished synchronously — return as-is
        return commit_res

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        await asyncio.sleep(0.5)
        try:
            task = await client.get_task(task_id)
        except Exception:
            continue
        if not isinstance(task, dict):
            continue
        status = task.get("status")
        if status == "completed":
            tr = task.get("result", {}) or {}
            commit_res["status"] = "completed"
            commit_res["memories_extracted"] = tr.get("memories_extracted", {})
            return commit_res
        if status == "failed":
            commit_res["status"] = "failed"
            commit_res["error"] = task.get("error", "unknown")
            return commit_res

    commit_res["status"] = "timeout"
    return commit_res


async def ingest_session(
    sess: Dict,
    sample_id: str,
    commit_timeout_s: int,
) -> Dict[str, Any]:
    """Ingest one LoCoMo session via the plugin-style flow.

    Each session uses its OWN client + lifecycle, mirroring how
    context-engine.ts creates a fresh OV session per turn-batch.
    """
    t0 = time.perf_counter()
    client = _make_client()
    await client.initialize()

    try:
        create_res = await client.create_session()
        session_id = create_res["session_id"]

        # Optional session-time anchor: per-message created_at = base + offset(seconds)
        # so the extractor's temporal reasoning gets stable absolute timestamps.
        base_dt: Optional[datetime] = None
        if sess["date_time"]:
            try:
                base_dt = datetime.strptime(sess["date_time"], "%I:%M %p on %d %B, %Y")
            except ValueError:
                base_dt = None

        for idx, msg in enumerate(sess["messages"]):
            msg_time = (base_dt + timedelta(seconds=idx)).isoformat() if base_dt else None
            await client.add_message(
                session_id=session_id,
                role=msg["role"],
                parts=[{"type": "text", "text": msg["text"]}],
                created_at=msg_time,
            )

        commit_res = await _commit_with_polling(client, session_id, commit_timeout_s)
        elapsed = time.perf_counter() - t0

        telemetry = (commit_res.get("telemetry", {}) or {}).get("summary", {}) or {}
        tokens = (telemetry.get("tokens", {}) or {}).get("llm", {}) or {}

        memories_extracted = commit_res.get("memories_extracted", {}) or {}
        total_mem = sum(memories_extracted.values()) if memories_extracted else 0

        return {
            "sample_id": sample_id,
            "session_key": sess["session_key"],
            "session_id": session_id,
            "message_count": len(sess["messages"]),
            "status": commit_res.get("status", "?"),
            "time_s": round(elapsed, 2),
            "memories_extracted": memories_extracted,
            "memories_total": total_mem,
            "llm_tokens": tokens.get("total", 0),
            "error": commit_res.get("error"),
        }

    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "sample_id": sample_id,
            "session_key": sess["session_key"],
            "status": "exception",
            "time_s": round(elapsed, 2),
            "error": str(e),
        }
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Inventory (post-ingest dump for sanity)
# ---------------------------------------------------------------------------
async def get_memory_inventory() -> Dict[str, List[Dict]]:
    """Recursively list extracted memories under the bench tenant."""
    client = _make_client()
    await client.initialize()

    inventory: Dict[str, List[Dict]] = {}
    try:
        for prefix in [
            f"viking://user/{BENCH_USER}/memories",
            "viking://agent/memories",
        ]:
            try:
                entries = await client.ls(prefix, recursive=True)
            except Exception:
                continue
            for e in entries:
                if e.get("isDir") or not e.get("name", "").endswith(".md"):
                    continue
                name = e.get("name", "")
                if name.startswith("."):
                    continue
                uri = e.get("uri", "")
                parts = uri.split("/")
                cat = "unknown"
                for i, p in enumerate(parts):
                    if p == "memories" and i + 1 < len(parts):
                        cat = parts[i + 1]
                        break
                inventory.setdefault(cat, []).append({"name": name, "uri": uri})
    finally:
        await client.close()

    return inventory


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
async def cleanup() -> None:
    client = _make_client()
    await client.initialize()
    try:
        for prefix in [f"viking://user/{BENCH_USER}", "viking://agent"]:
            try:
                await client.rm(prefix, recursive=True)
                print(f"  Cleaned: {prefix}")
            except Exception as e:
                print(f"  {prefix}: {e}")
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_range(s: str) -> Tuple[int, int]:
    parts = s.split("-")
    if len(parts) == 1:
        v = int(parts[0])
        return (v, v)
    return (int(parts[0]), int(parts[1]))


async def run(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("  LoCoMo INGESTION (plugin-fidelity)")
    print(f"  Server: {SERVER_URL}")
    vlm = load_vlm_config()
    print_vlm_banner(vlm)
    print(f"  Tenant: account={BENCH_ACCOUNT}  user={BENCH_USER}  agent={BENCH_AGENT}")
    print(f"  Source: {args.locomo}")
    print("=" * 80)

    if args.clean:
        print("\n  Cleaning previous bench data...")
        await cleanup()

    sample = load_locomo(args.locomo, args.sample)
    sample_id = sample.get("sample_id", f"sample_{args.sample}")
    session_range = parse_range(args.sessions) if args.sessions else None
    sessions = get_sessions(sample, session_range)

    print(f"\n  Sample {sample_id}: {len(sessions)} sessions to ingest")
    if sessions:
        speakers = sessions[0].get("speakers", "")
        if speakers:
            print(f"  Speakers: {speakers}")

    # ── Ingest loop ──
    results: List[Dict] = []
    t_start = time.perf_counter()
    for sess in sessions:
        result = await ingest_session(sess, sample_id, args.commit_timeout)
        results.append(result)

        status = result.get("status", "?")
        n_msgs = result.get("message_count", 0)
        n_mem = result.get("memories_total", 0)
        elapsed = result.get("time_s", 0)
        suffix = ""
        if status not in ("completed", "?") and result.get("error"):
            suffix = f"  [{result['error'][:80]}]"
        print(
            f"    [{result['session_key']}] {status:>10}  "
            f"{elapsed:6.1f}s  {n_msgs:>3} msgs  {n_mem:>3} memories{suffix}"
        )

    total_elapsed = time.perf_counter() - t_start

    # ── Summary ──
    n_done = sum(1 for r in results if r.get("status") == "completed")
    n_timeout = sum(1 for r in results if r.get("status") == "timeout")
    n_failed = sum(1 for r in results if r.get("status") in ("failed", "exception"))
    total_mem = sum(r.get("memories_total", 0) for r in results)

    print(f"\n  Ingest complete in {total_elapsed:.1f}s")
    print(f"    completed: {n_done}/{len(results)}")
    print(f"    timeouts:  {n_timeout}")
    print(f"    failed:    {n_failed}")
    print(f"    memories from server telemetry: {total_mem}")

    # Inventory from filesystem (authoritative count)
    print("\n  Fetching memory inventory from server...")
    inventory = await get_memory_inventory()
    total_files = sum(len(v) for v in inventory.values())
    print(f"  Memory inventory ({total_files} leaf files):")
    for cat in sorted(inventory.keys()):
        print(f"    {cat:<15} {len(inventory[cat]):>4}")

    # ── Save report ──
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or (
        RESULT_DIR / f"ingest_{sample_id}_{timestamp}.json"
    )
    report = {
        "timestamp": timestamp,
        "server_url": SERVER_URL,
        "vlm": vlm,
        "tenant": {"account": BENCH_ACCOUNT, "user": BENCH_USER, "agent_id": BENCH_AGENT},
        "sample_id": sample_id,
        "sessions_ingested": len(results),
        "sessions_completed": n_done,
        "sessions_timeout": n_timeout,
        "sessions_failed": n_failed,
        "total_time_s": round(total_elapsed, 2),
        "memories_total_telemetry": total_mem,
        "memories_total_filesystem": total_files,
        "inventory_by_category": {k: len(v) for k, v in sorted(inventory.items())},
        "sessions": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoCoMo ingestion script (plugin-fidelity, no QA)",
    )
    parser.add_argument("--locomo", default=LOCOMO_PATH, help="Path to locomo10.json")
    parser.add_argument("--sample", type=int, default=0, help="LoCoMo sample index")
    parser.add_argument("--sessions", default=None, help="Session range, e.g. '1-5' or '3'")
    parser.add_argument(
        "--commit-timeout", type=int, default=DEFAULT_COMMIT_TIMEOUT_S,
        help="Per-session commit-poll deadline in seconds",
    )
    parser.add_argument("--clean", action="store_true", help="Wipe bench tenant first")
    parser.add_argument("--output", default=None, help="Path for JSON summary report")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
