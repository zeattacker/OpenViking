"""Diagnose zero-recall questions. Dumps gold vs retrieved for first N failures."""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_recall_r10_locomo import (  # type: ignore
    API_BASE,
    API_KEY,
    ACCOUNT,
    USER,
    AGENT,
    build_dialog_session_map,
    evidence_to_session_ids,
    load_locomo,
    memory_to_sessions,
    ov_find,
)


def fetch_all_memories() -> list[dict]:
    """List all memories via find with empty query + big limit (best-effort)."""
    url = f"{API_BASE}/api/v1/search/find"
    body = json.dumps({"query": "the", "limit": 1000}).encode()
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
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read()).get("result", {}).get("memories", [])


def main():
    sample = load_locomo("/tmp/locomo/data/locomo10.json", 0)
    dialog_map = build_dialog_session_map(sample)
    qa_items = sample["qa"]

    # Stats on full inventory
    print("── INVENTORY ──")
    all_mems = fetch_all_memories()
    print(f"Total memories returned by broad query: {len(all_mems)}")
    by_session: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for m in all_mems:
        sessions = memory_to_sessions(m, dialog_map)
        for s in sessions:
            by_session[s] = by_session.get(s, 0) + 1
        # Extract memory type from URI
        uri = m.get("uri", "")
        type_match = re.search(r"memories/(\w+)/", uri)
        if type_match:
            t = type_match.group(1)
            by_type[t] = by_type.get(t, 0) + 1
    print(f"Memories per session: {dict(sorted(by_session.items()))}")
    print(f"Memories per type:    {by_type}")
    print(f"Total sessions with ANY memory: {len(by_session)} / 19")
    print()

    # Find zero-recall cases at limit=30
    print("── ZERO-RECALL ANALYSIS (limit=30) ──")
    zero_cases = []
    for i, qa in enumerate(qa_items):
        gold = evidence_to_session_ids(qa.get("evidence", []))
        if not gold:
            continue
        mems = ov_find(qa["question"], 30)
        retrieved = set()
        for m in mems:
            retrieved |= memory_to_sessions(m, dialog_map)
        hits = len(gold & retrieved)
        if hits == 0:
            zero_cases.append((i, qa, gold, retrieved, mems))

    print(f"Zero-recall questions at limit=30: {len(zero_cases)}")
    print()

    # Dump first 5 cases
    for idx, (i, qa, gold, retrieved, mems) in enumerate(zero_cases[:5]):
        print(f"── CASE {idx + 1} (q#{i}) ──")
        print(f"Q: {qa['question']}")
        print(f"Gold answer: {qa.get('answer', '?')}")
        print(f"Evidence: {qa.get('evidence')}")
        print(f"Gold sessions: {sorted(gold)}")
        print(f"Retrieved sessions: {sorted(retrieved)}")
        print(f"Category: {qa.get('category')}")
        # Check if gold session has ANY memory at all
        for gs in gold:
            cnt = by_session.get(gs, 0)
            print(f"  {gs}: {cnt} memories in full inventory")
        print(f"Top-5 memories returned:")
        for m in mems[:5]:
            score = m.get("score", 0)
            uri = m.get("uri", "?")
            abstract = (m.get("abstract") or "").replace("\n", " ")[:120]
            print(f"  [{score:.3f}] {uri}")
            print(f"           {abstract}")
        print()


if __name__ == "__main__":
    main()
