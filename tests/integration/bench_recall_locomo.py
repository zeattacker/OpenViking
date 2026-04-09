#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
LoCoMo Recall + QA Benchmark (plugin-fidelity).

Mirrors how openclaw-plugin's memory_recall tool fetches memories at QA time:

  for each question:
      1. POST /api/v1/search/search (via SDK .search()) in PARALLEL for both:
           - viking://user/{user}/memories
           - viking://agent/memories
         using requestLimit = max(limit * 4, 20), score_threshold = 0
      2. merge + dedupe by uri
      3. filter to level == 2 (leaf files only)
      4. expand entity links (1-hop, max 3 linked entities) — port of
         expandEntityLinks() in openclaw-plugin/index.ts:121
      5. read leaf contents → build context string
      6. run QA model with that context
      7. judge with the larger model

This script does NO ingestion. Run bench_ingest_locomo.py first.

Single-model: the QA model + judge model are BOTH read from ov.conf's `vlm`
section. To benchmark a different model, change ov.conf (and restart OV if
api_base changes), re-ingest, then run this script again. Output filenames
include the model slug so multiple runs don't collide.

Usage:
    cd source/
    source .venv/bin/activate
    python tests/integration/bench_recall_locomo.py            # full sample 0 QA
    python tests/integration/bench_recall_locomo.py --count 10 # smoke test
    python tests/integration/bench_recall_locomo.py --recall-limit 15 --no-expand
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure source root is on path
_source_root = str(Path(__file__).resolve().parent.parent.parent)
if _source_root not in sys.path:
    sys.path.insert(0, _source_root)

try:
    import openai as openai_lib
except ImportError:
    print("ERROR: pip install openai", file=sys.stderr)
    sys.exit(1)

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

QA_CATEGORIES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}


# ---------------------------------------------------------------------------
# ov.conf-driven VLM (single source of truth — same model OV uses)
# ---------------------------------------------------------------------------
def load_vlm_config() -> Dict[str, Any]:
    """Read ov.conf vlm section.

    The container mounts ov.conf as a bind volume so reading from the host
    sees the exact same config the OpenViking server is using. We rewrite
    `host.docker.internal` → `127.0.0.1` so the host-side bench script can
    reach the same endpoint the in-container server hits.
    """
    try:
        with open(OV_CONF_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        raise SystemExit(f"ERROR: cannot read {OV_CONF_PATH}: {e}")
    vlm = cfg.get("vlm") or {}
    if not vlm.get("model") or not vlm.get("api_base"):
        raise SystemExit(
            f"ERROR: ov.conf vlm section missing model/api_base: {vlm}"
        )
    api_base = vlm.get("api_base", "")
    # docker→host rewrite so the script (running on the host) can reach the
    # same endpoint that the in-container OV server uses.
    vlm["api_base"] = api_base.replace("host.docker.internal", "127.0.0.1")
    return vlm


def print_vlm_banner(vlm: Dict[str, Any]) -> None:
    print(f"  VLM (extraction + QA + judge): {vlm.get('model', '?')}")
    print(f"    api_base:    {vlm.get('api_base', '?')}")
    print(f"    provider:    {vlm.get('provider', '?')}  "
          f"temperature: {vlm.get('temperature', '?')}  "
          f"thinking: {vlm.get('thinking', False)}")

# Mirror plugin's plugin/index.ts:115
ENTITY_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)")


# ---------------------------------------------------------------------------
# OpenViking client
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
def load_qa_items(path: str, sample_index: int, count: Optional[int]) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if sample_index < 0 or sample_index >= len(data):
        raise ValueError(f"sample_index {sample_index} out of range (0-{len(data) - 1})")
    sample = data[sample_index]
    items: List[Dict] = []
    for qa in sample.get("qa", []):
        cat = qa.get("category", 0)
        items.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "category": cat,
            "category_name": QA_CATEGORIES.get(cat, "unknown"),
        })
    if count:
        items = items[:count]
    return items


# ---------------------------------------------------------------------------
# Plugin-fidelity recall
# ---------------------------------------------------------------------------
def _result_to_items(result: Any) -> List[Dict[str, Any]]:
    """Normalize FindResult / dict / list of memories into list of dicts."""
    if result is None:
        return []
    if hasattr(result, "memories"):
        memories = result.memories or []
    elif isinstance(result, dict):
        memories = result.get("memories", []) or []
    elif isinstance(result, list):
        memories = result
    else:
        return []
    out: List[Dict[str, Any]] = []
    for m in memories:
        if hasattr(m, "to_dict"):
            out.append(m.to_dict())
        elif hasattr(m, "__dict__"):
            d = {k: v for k, v in m.__dict__.items() if not k.startswith("_")}
            out.append(d)
        elif isinstance(m, dict):
            out.append(m)
    return out


async def _safe_search(
    client: _OVClient,
    query: str,
    target_uri: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Wrap client.search() with exception swallowing (mirrors Promise.allSettled)."""
    try:
        result = await client.search(
            query=query,
            target_uri=target_uri,
            limit=limit,
            score_threshold=0.0,
        )
        return _result_to_items(result)
    except Exception:
        return []


async def expand_entity_links(
    memories: List[Dict[str, Any]],
    client: _OVClient,
    max_expand: int = 3,
) -> List[Dict[str, Any]]:
    """Port of openclaw-plugin/index.ts:121 expandEntityLinks().

    For each retrieved memory, read its content, regex `[name](path.md)` links,
    resolve to absolute URIs, dedupe against existing, fetch up to max_expand
    new linked memories. Returns the expanded entries (callers concat with
    the originals).
    """
    if not memories or max_expand <= 0:
        return []

    existing_uris: Set[str] = {m.get("uri", "") for m in memories if m.get("uri")}
    linked_uris: List[str] = []
    seen_links: Set[str] = set()

    for mem in memories:
        uri = mem.get("uri", "")
        if not uri:
            continue
        try:
            content = await client.read(uri)
        except Exception:
            continue
        if not content or not isinstance(content, str):
            continue

        # Walk all entity links inside this memory's content
        mem_dir = uri[: uri.rfind("/")]
        parent_dir = mem_dir[: mem_dir.rfind("/")]
        for match in ENTITY_LINK_RE.finditer(content):
            linked_file = match.group(2)
            if linked_file.startswith("../"):
                entity_uri = f"{parent_dir}/{linked_file[3:]}"
            elif "/" in linked_file:
                entity_uri = f"{parent_dir}/{linked_file}"
            else:
                entity_uri = f"{mem_dir}/{linked_file}"

            if entity_uri in existing_uris or entity_uri in seen_links:
                continue
            seen_links.add(entity_uri)
            linked_uris.append(entity_uri)

        if len(linked_uris) >= max_expand:
            break

    expanded: List[Dict[str, Any]] = []
    for entity_uri in linked_uris[:max_expand]:
        try:
            content = await client.read(entity_uri)
        except Exception:
            continue
        if content and isinstance(content, str) and content.strip():
            expanded.append({
                "uri": entity_uri,
                "level": 2,
                "score": 0.5,
                "abstract": content.strip()[:200],
                "category": "entities",
                "_expanded": True,
                "_content": content,
            })
    return expanded


async def recall_for_question(
    client: _OVClient,
    query: str,
    limit: int,
    expand: bool,
) -> tuple[List[Dict[str, Any]], int]:
    """Mirror of plugin/index.ts:618-684 memory_recall flow.

    Returns (final_memories_with_content, total_after_expansion).
    """
    request_limit = max(limit * 4, 20)

    user_target = f"viking://user/{BENCH_USER}/memories"
    agent_target = "viking://agent/memories"

    user_mems, agent_mems = await asyncio.gather(
        _safe_search(client, query, user_target, request_limit),
        _safe_search(client, query, agent_target, request_limit),
    )

    # Merge and dedupe by uri
    seen: Set[str] = set()
    merged: List[Dict[str, Any]] = []
    for m in user_mems + agent_mems:
        u = m.get("uri", "")
        if not u or u in seen:
            continue
        seen.add(u)
        merged.append(m)

    # Plugin: filter to level == 2 (leaf files only — drop overviews/abstracts)
    leaf_only = [m for m in merged if m.get("level", 2) == 2]

    # Sort by score descending and apply final limit
    leaf_only.sort(key=lambda m: float(m.get("score", 0) or 0), reverse=True)
    top = leaf_only[:limit]

    # Read leaf contents (expand_entity_links also reads, but it dedupes)
    for m in top:
        if "_content" in m:
            continue
        try:
            c = await client.read(m["uri"])
            m["_content"] = c if isinstance(c, str) else ""
        except Exception:
            m["_content"] = ""

    # 1-hop entity expansion
    expanded: List[Dict[str, Any]] = []
    if expand:
        expanded = await expand_entity_links(top, client, max_expand=3)

    return top + expanded, len(top) + len(expanded)


def build_context_string(memories: List[Dict[str, Any]]) -> str:
    """Format recalled memories for the QA prompt."""
    parts: List[str] = []
    for m in memories:
        content = (m.get("_content") or "").strip()
        if not content:
            continue
        uri = m.get("uri", "")
        cat = "?"
        # Extract category from uri: viking://.../memories/{cat}/file.md
        toks = uri.split("/")
        for i, t in enumerate(toks):
            if t == "memories" and i + 1 < len(toks):
                cat = toks[i + 1]
                break
        name = uri.rsplit("/", 1)[-1].removesuffix(".md")
        marker = " [linked]" if m.get("_expanded") else ""
        parts.append(f"### [{cat}] {name}{marker}\n{content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# QA evaluation
# ---------------------------------------------------------------------------
QA_SYSTEM_TEMPLATE = """You are a helpful assistant with memory of past conversations between two friends.
Use ONLY the provided memories to answer. If the answer is not in the memories, say "I don't know."
Keep answers concise — 1-2 sentences max.

## Memories:
{context}
"""


async def eval_one_question(
    llm_client: openai_lib.AsyncOpenAI,
    vlm: Dict[str, Any],
    ov_client: _OVClient,
    qa: Dict,
    recall_limit: int,
    expand: bool,
) -> Dict[str, Any]:
    """Per-question recall + answer."""
    t_recall = time.perf_counter()
    memories, n_recalled = await recall_for_question(
        ov_client, qa["question"], recall_limit, expand=expand,
    )
    recall_s = time.perf_counter() - t_recall

    context = build_context_string(memories)
    system = QA_SYSTEM_TEMPLATE.format(context=context if context else "(no relevant memories)")

    t_llm = time.perf_counter()
    try:
        resp = await llm_client.chat.completions.create(
            model=vlm["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": qa["question"]},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        llm_s = time.perf_counter() - t_llm
        answer = (resp.choices[0].message.content or "").strip()
        tokens = resp.usage.total_tokens if resp.usage else 0
        err = None
    except Exception as e:
        llm_s = time.perf_counter() - t_llm
        answer = f"[ERROR] {e}"
        tokens = 0
        err = str(e)

    return {
        "question": qa["question"],
        "gold_answer": qa["answer"],
        "response": answer,
        "category": qa["category"],
        "category_name": qa["category_name"],
        "recall_s": round(recall_s, 3),
        "llm_s": round(llm_s, 3),
        "time_s": round(recall_s + llm_s, 3),
        "tokens": tokens,
        "memories_recalled": n_recalled,
        "memory_context_chars": len(context),
        "error": err,
    }


async def eval_with_vlm(
    vlm: Dict[str, Any],
    qa_items: List[Dict],
    recall_limit: int,
    expand: bool,
) -> List[Dict]:
    """Run QA over all questions using the ov.conf VLM."""
    llm_client = openai_lib.AsyncOpenAI(
        api_key=vlm.get("api_key") or "not-needed",
        base_url=vlm["api_base"],
    )
    ov_client = _make_client()
    await ov_client.initialize()

    print(f"\n  --- QA with {vlm['model']} ---")
    results: List[Dict] = []
    try:
        for i, qa in enumerate(qa_items):
            r = await eval_one_question(llm_client, vlm, ov_client, qa, recall_limit, expand)
            results.append(r)
            if (i + 1) % 5 == 0 or i == len(qa_items) - 1:
                print(f"    {i + 1}/{len(qa_items)} done")
    finally:
        await ov_client.close()
        await llm_client.close()
    return results


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """Grade if the generated answer matches the gold answer. Be generous — same topic/fact = CORRECT.

Question: {question}
Gold answer: {gold}
Generated answer: {response}

Respond JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "brief"}}"""


async def judge_results(results: List[Dict], vlm: Dict[str, Any]) -> List[Dict]:
    """Judge using the same VLM that answers (single source of truth from ov.conf)."""
    client = openai_lib.AsyncOpenAI(
        api_key=vlm.get("api_key") or "not-needed",
        base_url=vlm["api_base"],
    )
    correct = 0
    try:
        for i, r in enumerate(results):
            if r["response"].startswith("[ERROR]"):
                r["grade"] = "ERROR"
                r["reasoning"] = r.get("error", "")
                continue
            try:
                resp = await client.chat.completions.create(
                    model=vlm["model"],
                    messages=[
                        {"role": "system", "content": "You are an expert grader. JSON only."},
                        {
                            "role": "user",
                            "content": JUDGE_PROMPT.format(
                                question=r["question"],
                                gold=r["gold_answer"],
                                response=r["response"],
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=150,
                )
                content = resp.choices[0].message.content or ""
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    j = json.loads(content[start:end + 1])
                    r["grade"] = "CORRECT" if str(j.get("is_correct", "")).upper() == "CORRECT" else "WRONG"
                    r["reasoning"] = j.get("reasoning", "")
                else:
                    r["grade"] = "PARSE_ERROR"
                    r["reasoning"] = content[:200]
            except Exception as e:
                r["grade"] = "JUDGE_ERROR"
                r["reasoning"] = str(e)

            if r.get("grade") == "CORRECT":
                correct += 1
            if (i + 1) % 10 == 0 or i == len(results) - 1:
                print(f"    Judged {i + 1}/{len(results)}: {correct}/{i + 1} correct")
    finally:
        await client.close()
    return results


# ---------------------------------------------------------------------------
# Reporting + persistence
# ---------------------------------------------------------------------------
def compute_metrics(graded: List[Dict]) -> Dict[str, Any]:
    by_cat: Dict[str, Dict[str, int]] = {}
    total_c = total_n = 0
    total_t = 0.0
    total_recall = 0.0
    total_mem = 0
    for r in graded:
        cat = r.get("category_name", "unknown")
        by_cat.setdefault(cat, {"correct": 0, "total": 0})
        by_cat[cat]["total"] += 1
        total_n += 1
        total_t += r.get("time_s", 0)
        total_recall += r.get("recall_s", 0)
        total_mem += r.get("memories_recalled", 0)
        if r.get("grade") == "CORRECT":
            by_cat[cat]["correct"] += 1
            total_c += 1
    return {
        "accuracy": total_c / max(total_n, 1),
        "correct": total_c,
        "total": total_n,
        "avg_time_s": total_t / max(total_n, 1),
        "avg_recall_s": total_recall / max(total_n, 1),
        "avg_memories_recalled": total_mem / max(total_n, 1),
        "by_category": {
            cat: {"accuracy": d["correct"] / max(d["total"], 1), **d}
            for cat, d in sorted(by_cat.items())
        },
    }


def save_csv(graded: List[Dict], path: Path) -> None:
    fields = [
        "question", "gold_answer", "response", "grade", "reasoning",
        "category", "category_name", "memories_recalled", "memory_context_chars",
        "recall_s", "llm_s", "time_s", "tokens",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in graded:
            w.writerow({k: r.get(k, "") for k in fields})


def print_full_report(model: str, m: Dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print(f"  LoCoMo RECALL BENCHMARK — {model}")
    print("=" * 90)
    print(f"\n  Overall accuracy: {m['accuracy']:.1%}  ({m['correct']}/{m['total']})")
    print(f"  Avg total time:   {m['avg_time_s']:.2f}s")
    print(f"  Avg recall time:  {m['avg_recall_s']:.2f}s")
    print(f"  Avg memories recalled per question: {m['avg_memories_recalled']:.1f}")
    print(f"\n  {'CATEGORY':<20} {'ACC':>8}  {'CORRECT/TOTAL':>15}")
    print(f"  {'-' * 20} {'-' * 8}  {'-' * 15}")
    for cat, d in m["by_category"].items():
        ratio = f"{d['correct']}/{d['total']}"
        print(f"  {cat:<20} {d['accuracy']:>7.1%}  {ratio:>15}")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def assert_ingestion_exists() -> None:
    """Crash early if memory tree is empty — forces user to run ingest first."""
    client = _make_client()
    await client.initialize()
    try:
        try:
            entries = await client.ls(
                f"viking://user/{BENCH_USER}/memories",
                recursive=True,
            )
        except Exception:
            entries = []
        files = [e for e in entries if not e.get("isDir") and not e.get("name", "").startswith(".")]
        if len(files) < 5:
            raise SystemExit(
                f"\nERROR: bench tenant has only {len(files)} memory files.\n"
                f"Run bench_ingest_locomo.py first to populate viking://user/{BENCH_USER}/memories\n"
            )
        print(f"  Ingestion check: {len(files)} memory files found in bench tenant")
    finally:
        await client.close()


async def run(args: argparse.Namespace) -> None:
    vlm = load_vlm_config()
    print("=" * 80)
    print("  LoCoMo RECALL + QA BENCHMARK (plugin-fidelity)")
    print(f"  Server: {SERVER_URL}")
    print_vlm_banner(vlm)
    print(f"  Tenant: account={BENCH_ACCOUNT} user={BENCH_USER}")
    print(f"  Recall limit: {args.recall_limit}  Entity expansion: {'on' if not args.no_expand else 'off'}")
    print("=" * 80)

    await assert_ingestion_exists()

    qa_items = load_qa_items(args.locomo, args.sample, args.count)
    print(f"  Loaded {len(qa_items)} QA items from sample {args.sample}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── QA pass ──
    results = await eval_with_vlm(vlm, qa_items, args.recall_limit, expand=not args.no_expand)

    # ── Judge pass (same VLM) ──
    print(f"\n  Judging with {vlm['model']}...")
    graded = await judge_results(results, vlm)
    metrics = compute_metrics(graded)

    # Use a filesystem-safe slug from the model name for output filenames
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", vlm["model"]).strip("_") or "model"
    csv_path = RESULT_DIR / f"recall_{model_slug}_{timestamp}.csv"
    save_csv(graded, csv_path)
    print(f"  Saved CSV: {csv_path}")

    print_full_report(vlm["model"], metrics)

    summary_path = RESULT_DIR / f"recall_{model_slug}_{timestamp}.json"
    summary = {
        "timestamp": timestamp,
        "server_url": SERVER_URL,
        "vlm": vlm,
        "tenant": {"account": BENCH_ACCOUNT, "user": BENCH_USER},
        "sample": args.sample,
        "qa_count": len(qa_items),
        "recall_limit": args.recall_limit,
        "entity_expansion": not args.no_expand,
        "metrics": metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary: {summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="LoCoMo recall + QA bench (plugin-fidelity, no ingestion)",
    )
    p.add_argument("--locomo", default=LOCOMO_PATH, help="Path to locomo10.json")
    p.add_argument("--sample", type=int, default=0, help="LoCoMo sample index for QA")
    p.add_argument("--count", type=int, default=None, help="Limit number of QA items")
    p.add_argument(
        "--recall-limit", type=int, default=10,
        help="Per-question top-k after dedupe/level-2/postprocess (plugin default = 10)",
    )
    p.add_argument(
        "--no-expand", action="store_true",
        help="Disable expand_entity_links (1-hop link follow)",
    )
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
