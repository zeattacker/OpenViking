"""
OpenViking retrieval-only R@10 on LoCoMo — apples-to-apples with MemPalace.

Measures the SAME metric MemPalace's benchmarks/locomo_bench.py reports:
    recall = |gold_session_ids ∩ retrieved_session_ids| / |gold_session_ids|

Pipeline:
    1. Load LoCoMo conv-26 (already ingested into OpenViking by bench_ingest_locomo)
    2. Build dialog_text → session_id map
    3. For each of 199 questions, call /api/v1/search/find with limit=N_memories
    4. Parse each returned memory's source dialog to determine which sessions it covers
    5. Compute recall against evidence_to_session_ids(qa["evidence"])

This isolates OpenViking's retrieval from its answer generator.
No LLM calls. No VLM. Pure vector search + text overlap for session mapping.

Usage:
    python tests/integration/bench_recall_r10_locomo.py
    python tests/integration/bench_recall_r10_locomo.py --memory-limit 30
    python tests/integration/bench_recall_r10_locomo.py --search  # use /search not /find
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import urllib.request

API_BASE = os.environ.get("OPENVIKING_API_BASE", "http://127.0.0.1:1933")
API_KEY = os.environ.get("OPENVIKING_API_KEY", "openviking-local")
ACCOUNT = os.environ.get("OPENVIKING_ACCOUNT", "bench")
USER = os.environ.get("OPENVIKING_USER", "bench-locomo")
AGENT = os.environ.get("OPENVIKING_AGENT", "bench-locomo")

RESULT_DIR = Path(__file__).parent / "bench_results"
CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}


def load_locomo(path: str, sample_index: int = 0) -> dict:
    data = json.load(open(path))
    return data[sample_index]


def build_dialog_session_map(sample: dict) -> dict[str, str]:
    """Return {normalized_dialog_text: session_id}. Text is lowercased + whitespace-collapsed."""
    conv = sample["conversation"]
    dialog_map: dict[str, str] = {}
    session_keys = sorted(
        [k for k in conv.keys() if re.fullmatch(r"session_\d+", k)],
        key=lambda k: int(k.split("_")[1]),
    )
    for skey in session_keys:
        turns = conv[skey]
        if not isinstance(turns, list):
            continue
        for d in turns:
            text = (d.get("text") or "").strip()
            if len(text) < 15:
                continue
            norm = re.sub(r"\s+", " ", text.lower())
            # Only keep the first occurrence — dialogs may repeat across sessions
            dialog_map.setdefault(norm, skey)
    return dialog_map


def evidence_to_session_ids(evidence: list[str]) -> set[str]:
    """D1:3 → session_1 (MemPalace's exact logic)."""
    sessions = set()
    for eid in evidence or []:
        m = re.match(r"D(\d+):", eid)
        if m:
            sessions.add(f"session_{m.group(1)}")
    return sessions


def memory_to_sessions(memory: dict, dialog_map: dict[str, str]) -> set[str]:
    """Find which sessions a memory's source text covers via substring matching."""
    haystack = " ".join(
        [
            memory.get("abstract") or "",
            memory.get("overview") or "",
        ]
    )
    norm = re.sub(r"\s+", " ", haystack.lower())
    covered: set[str] = set()
    for text, sess_id in dialog_map.items():
        if text in norm:
            covered.add(sess_id)
    return covered


def ov_find(query: str, limit: int, endpoint: str = "find") -> list[dict[str, Any]]:
    url = f"{API_BASE}/api/v1/search/{endpoint}"
    body = json.dumps({"query": query, "limit": limit}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
            "X-OpenViking-Account": ACCOUNT,
            "X-OpenViking-User": USER,
            "X-OpenViking-Agent": AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read())
    except Exception as e:
        print(f"  [WARN] find failed: {e}")
        return []
    if payload.get("status") != "ok":
        return []
    result = payload.get("result") or {}
    mems = result.get("memories") or []
    # Sort by score desc just in case
    mems.sort(key=lambda m: -(m.get("score") or 0))
    return mems


def run(args: argparse.Namespace) -> None:
    sample = load_locomo(args.data_file, args.sample)
    sample_id = sample.get("sample_id", f"sample_{args.sample}")
    qa_items = sample.get("qa", [])
    if args.count:
        qa_items = qa_items[: args.count]

    dialog_map = build_dialog_session_map(sample)
    print(f"  Sample: {sample_id}  |  QAs: {len(qa_items)}  |  Dialogs in map: {len(dialog_map)}")
    print(f"  Endpoint: /api/v1/search/{args.endpoint}  |  Memory limit: {args.memory_limit}")
    print("─" * 60)

    per_cat: dict[str, list[float]] = defaultdict(list)
    all_recalls: list[float] = []
    mem_cover_sizes: list[int] = []
    t0 = time.perf_counter()
    empty_results = 0
    zero_gold = 0

    for i, qa in enumerate(qa_items):
        question = qa.get("question") or ""
        evidence = qa.get("evidence") or []
        cat = CATEGORY_NAMES.get(qa.get("category") or 0, "unknown")
        gold_sessions = evidence_to_session_ids(evidence)
        if not gold_sessions:
            zero_gold += 1
            continue

        mems = ov_find(question, args.memory_limit, args.endpoint)
        if not mems:
            empty_results += 1

        retrieved_sessions: set[str] = set()
        for m in mems:
            retrieved_sessions |= memory_to_sessions(m, dialog_map)
        mem_cover_sizes.append(len(retrieved_sessions))

        hits = len(gold_sessions & retrieved_sessions)
        recall = hits / len(gold_sessions) if gold_sessions else 0.0
        all_recalls.append(recall)
        per_cat[cat].append(recall)

        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t0
            running = sum(all_recalls) / len(all_recalls) if all_recalls else 0
            print(
                f"  [{i + 1:3}/{len(qa_items)}] running R={running:.3f}  "
                f"(elapsed {elapsed:.1f}s, {elapsed / (i + 1):.2f}s/q)"
            )

    total_time = time.perf_counter() - t0
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    avg_cover = sum(mem_cover_sizes) / len(mem_cover_sizes) if mem_cover_sizes else 0
    perfect = sum(1 for r in all_recalls if r >= 1.0)
    partial = sum(1 for r in all_recalls if 0 < r < 1.0)
    zero = sum(1 for r in all_recalls if r == 0)

    print()
    print("=" * 60)
    print(f"  RESULTS — OpenViking retrieval-only R@10 (session granularity)")
    print("=" * 60)
    print(f"  Sample:       {sample_id}")
    print(f"  Questions:    {len(all_recalls)}  (skipped {zero_gold} with empty evidence)")
    print(f"  Memory limit: {args.memory_limit}  (avg sessions covered: {avg_cover:.1f})")
    print(f"  Empty finds:  {empty_results}")
    print(f"  Time:         {total_time:.1f}s  ({total_time / max(1, len(all_recalls)):.2f}s/q)")
    print(f"  Avg Recall:   {avg_recall:.3f}")
    print()
    print("  PER-CATEGORY RECALL:")
    for cat in ["single-hop", "multi-hop", "temporal", "open-domain", "adversarial"]:
        vals = per_cat.get(cat, [])
        if vals:
            print(f"    {cat:<25} R={sum(vals) / len(vals):.3f}  (n={len(vals)})")
    print()
    print("  RECALL DISTRIBUTION:")
    n = max(1, len(all_recalls))
    print(f"    Perfect (1.0):  {perfect:4} ({perfect / n * 100:.1f}%)")
    print(f"    Partial (0-1):  {partial:4} ({partial / n * 100:.1f}%)")
    print(f"    Zero (0.0):     {zero:4} ({zero / n * 100:.1f}%)")
    print("=" * 60)

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = RESULT_DIR / f"recall_r10_{sample_id}_{args.endpoint}_lim{args.memory_limit}_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "sample_id": sample_id,
                "endpoint": args.endpoint,
                "memory_limit": args.memory_limit,
                "questions": len(all_recalls),
                "skipped_no_evidence": zero_gold,
                "empty_finds": empty_results,
                "avg_recall": avg_recall,
                "avg_sessions_covered": avg_cover,
                "time_s": total_time,
                "per_category": {k: sum(v) / len(v) for k, v in per_cat.items() if v},
                "per_category_n": {k: len(v) for k, v in per_cat.items()},
                "distribution": {"perfect": perfect, "partial": partial, "zero": zero},
            },
            indent=2,
        )
    )
    print(f"\n  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenViking retrieval-only R@10 on LoCoMo")
    parser.add_argument(
        "--data-file", default="/tmp/locomo/data/locomo10.json", help="LoCoMo JSON path"
    )
    parser.add_argument("--sample", type=int, default=0, help="Sample index (0=conv-26)")
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=10,
        help="Top-N memories per query (10 matches MemPalace budget)",
    )
    parser.add_argument(
        "--endpoint",
        choices=["find", "search"],
        default="find",
        help="/api/v1/search/find (no session) or /search (with session context)",
    )
    parser.add_argument("--count", type=int, default=0, help="Limit to first N questions")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
