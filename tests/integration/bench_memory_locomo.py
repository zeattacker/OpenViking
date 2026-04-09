#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
LoCoMo-based Memory Benchmark for OpenViking v2 — Qwen3.5-35B-A3B vs Bonsai-8B

Tests memory extraction quality (entities, preferences, events) and recall accuracy.
Data isolation: dedicated bench account/user, no pollution of existing memories.

Usage:
    cd source/
    source .venv/bin/activate

    # Full benchmark: import + wait + eval
    python tests/integration/bench_memory_locomo.py full --sample 0 --sessions 1-5 --count 20 --wait 60

    # Import only (server VLM extracts memories)
    python tests/integration/bench_memory_locomo.py import --sample 0 --sessions 1-3

    # Eval only (uses already-extracted memories, compares both models)
    python tests/integration/bench_memory_locomo.py eval --sample 0 --count 20

    # Cleanup bench data
    python tests/integration/bench_memory_locomo.py cleanup
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import openai as openai_lib
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# Ensure source root is on path
_source_root = str(Path(__file__).resolve().parent.parent.parent)
if _source_root not in sys.path:
    sys.path.insert(0, _source_root)

try:
    from openviking_cli.client.http import AsyncHTTPClient as _OVClient
except ImportError:
    _OVClient = None
    print("WARNING: openviking SDK not available", file=sys.stderr)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERVER_URL = "http://127.0.0.1:1933"
API_KEY = "openviking-local"
LOCOMO_PATH = os.path.expanduser("~/Documents/Projects/openclaw/openclaw-eval/locomo10.json")
RESULT_DIR = Path(__file__).parent / "bench_results"

# Single bench identity — both models eval against the SAME extracted memories
BENCH_ACCOUNT = "bench"
BENCH_USER = "bench-locomo"
BENCH_AGENT = "bench-locomo"

MODELS = {
    "qwen": {
        "api_base": "http://localhost:1234/v1",
        "api_key": "not-needed",
        "model": "Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-Q4_K_M.gguf",
        "label": "Qwen3.5-35B-A3B",
    },
    "bonsai": {
        "api_base": "http://localhost:8080/v1",
        "api_key": "not-needed",
        "model": "Bonsai-8B.gguf",
        "label": "Bonsai-8B",
    },
}

# LoCoMo QA categories
QA_CATEGORIES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}

# Target memory types for this benchmark
TARGET_TYPES = {"entities", "preferences", "events", "cases", "patterns"}


# ---------------------------------------------------------------------------
# OpenViking client helper
# ---------------------------------------------------------------------------
def _make_client():
    if _OVClient is None:
        raise RuntimeError("openviking SDK not available")
    return _OVClient(
        url=SERVER_URL,
        api_key=API_KEY,
        account=BENCH_ACCOUNT,
        user=BENCH_USER,
        agent_id=BENCH_AGENT,
    )


# ---------------------------------------------------------------------------
# LoCoMo data loading
# ---------------------------------------------------------------------------
def load_locomo(path: str, sample_index: Optional[int] = None) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if sample_index is not None:
        return [data[sample_index]]
    return data


def get_sessions(sample: Dict, session_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
    """Extract sessions mapped to user/assistant roles for proper memory extraction."""
    conv = sample["conversation"]
    speaker_a = conv["speaker_a"]  # → user
    speaker_b = conv["speaker_b"]  # → assistant

    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    sessions = []
    for sk in session_keys:
        num = int(sk.split("_")[1])
        if session_range and (num < session_range[0] or num > session_range[1]):
            continue

        dt_key = f"{sk}_date_time"
        date_time = conv.get(dt_key, "")

        messages = []
        for msg in conv[sk]:
            # Map speakers to user/assistant roles for memory extraction
            role = "user" if msg["speaker"] == speaker_a else "assistant"
            messages.append({"role": role, "text": msg["text"]})

        sessions.append({
            "session_key": sk,
            "date_time": date_time,
            "messages": messages,
            "speakers": f"{speaker_a} (user) / {speaker_b} (assistant)",
        })

    return sessions


def get_qa_items(sample: Dict, count: Optional[int] = None) -> List[Dict]:
    qa_list = []
    for qa in sample.get("qa", []):
        qa_list.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "category": qa.get("category", 0),
            "category_name": QA_CATEGORIES.get(qa.get("category", 0), "unknown"),
        })
    if count:
        qa_list = qa_list[:count]
    return qa_list


# ---------------------------------------------------------------------------
# Phase 1: Import
# ---------------------------------------------------------------------------
async def import_sessions(sessions: List[Dict], sample_id: str) -> List[Dict]:
    """Import LoCoMo sessions into OpenViking as user/assistant conversations."""
    results = []

    for sess in sessions:
        t0 = time.perf_counter()
        client = _make_client()
        await client.initialize()

        try:
            res = await client.create_session()
            session_id = res["session_id"]

            base_dt = None
            if sess["date_time"]:
                try:
                    base_dt = datetime.strptime(sess["date_time"], "%I:%M %p on %d %B, %Y")
                except ValueError:
                    pass

            for idx, msg in enumerate(sess["messages"]):
                msg_time = (base_dt + timedelta(seconds=idx)).isoformat() if base_dt else None
                await client.add_message(
                    session_id=session_id,
                    role=msg["role"],
                    parts=[{"type": "text", "text": msg["text"]}],
                    created_at=msg_time,
                )

            commit_res = await client.commit_session(session_id, telemetry=True)

            # Client-side task polling (mirrors openclaw-eval OV.commit wait=True).
            # The SDK's commit_session returns immediately with a task_id; we poll
            # /tasks/{task_id} until extraction completes so per-session memories
            # are ready before we move on.
            task_id = commit_res.get("task_id") if isinstance(commit_res, dict) else None
            if task_id:
                deadline = time.perf_counter() + 300  # 5-minute per-session cap
                while time.perf_counter() < deadline:
                    await asyncio.sleep(0.5)
                    try:
                        task = await client.get_task(task_id)
                    except Exception:
                        continue
                    status = task.get("status") if isinstance(task, dict) else None
                    if status == "completed":
                        tr = task.get("result", {}) or {}
                        commit_res["status"] = "completed"
                        commit_res["memories_extracted"] = tr.get("memories_extracted", {})
                        break
                    if status == "failed":
                        commit_res["status"] = "failed"
                        commit_res["error"] = task.get("error", "unknown")
                        break
                else:
                    commit_res["status"] = "timeout"
            elapsed = time.perf_counter() - t0

            telemetry = commit_res.get("telemetry", {}).get("summary", {})
            tokens = telemetry.get("tokens", {})

            result = {
                "session_key": sess["session_key"],
                "message_count": len(sess["messages"]),
                "status": commit_res.get("status", "?"),
                "time_s": round(elapsed, 2),
                "llm_tokens": tokens.get("llm", {}).get("total", 0),
            }
            results.append(result)
            status = result["status"]
            print(f"    [{sess['session_key']}] {status} in {elapsed:.1f}s ({len(sess['messages'])} msgs)")

        except Exception as e:
            results.append({"session_key": sess["session_key"], "status": "error", "error": str(e)})
            print(f"    [{sess['session_key']}] ERROR: {e}")
        finally:
            await client.close()

    return results


async def wait_for_extraction(timeout_s: int = 60):
    """Wait for server to finish async memory extraction."""
    client = _make_client()
    await client.initialize()
    try:
        print(f"  Waiting up to {timeout_s}s for extraction to complete...")
        await client.wait_processed(timeout=timeout_s)
        print("  Extraction complete.")
    except Exception as e:
        print(f"  Wait returned: {e}")
    finally:
        await client.close()


async def get_memory_inventory() -> Dict[str, Any]:
    """Get detailed inventory of extracted memories by type."""
    client = _make_client()
    await client.initialize()

    inventory = {}  # type -> list of {name, uri, chars}
    try:
        for prefix in [
            f"viking://user/{BENCH_USER}/memories",
            "viking://agent/memories",
        ]:
            try:
                entries = await client.ls(prefix, recursive=True)
                for e in entries:
                    if e.get("isDir") or not e.get("name", "").endswith(".md"):
                        continue
                    name = e.get("name", "")
                    if name.startswith("."):
                        continue
                    uri = e.get("uri", "")
                    # Determine category
                    parts = uri.split("/")
                    cat = "unknown"
                    for i, p in enumerate(parts):
                        if p == "memories" and i + 1 < len(parts):
                            cat = parts[i + 1]
                            break
                    if cat not in inventory:
                        inventory[cat] = []
                    # Read content
                    content = ""
                    try:
                        content = await client.read(uri) or ""
                    except Exception:
                        pass
                    inventory[cat].append({"name": name, "uri": uri, "content": content})
            except Exception:
                pass
    finally:
        await client.close()

    return inventory


async def fetch_memory_context() -> str:
    """Fetch all non-episode memories as QA context string."""
    inventory = await get_memory_inventory()
    texts = []
    for cat, items in sorted(inventory.items()):
        if cat == "episodes":
            continue  # Skip episodes per user request
        for item in items:
            if item["content"].strip():
                texts.append(f"### [{cat}] {item['name']}\n{item['content'].strip()}")
    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Phase 2: QA Eval
# ---------------------------------------------------------------------------
async def eval_qa(qa_items: List[Dict], model_key: str, memory_context: str) -> List[Dict]:
    """Run QA evaluation: give model the memories, ask LoCoMo questions."""
    cfg = MODELS[model_key]
    client = openai_lib.AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])
    results = []

    system_prompt = f"""You are a helpful assistant with memory of past conversations between two friends.
Use ONLY the provided memories to answer. If the answer is not in the memories, say "I don't know."
Keep answers concise — 1-2 sentences max.

## Memories:
{memory_context}
"""

    for i, qa in enumerate(qa_items):
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": qa["question"]},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            elapsed = time.perf_counter() - t0
            answer = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0

            results.append({
                "question": qa["question"],
                "gold_answer": qa["answer"],
                "response": answer.strip(),
                "category": qa["category"],
                "category_name": qa["category_name"],
                "time_s": round(elapsed, 2),
                "tokens": tokens,
            })
        except Exception as e:
            results.append({
                "question": qa["question"],
                "gold_answer": qa["answer"],
                "response": f"[ERROR] {e}",
                "category": qa["category"],
                "category_name": qa["category_name"],
                "time_s": round(time.perf_counter() - t0, 2),
                "tokens": 0,
            })

        if (i + 1) % 5 == 0 or i == len(qa_items) - 1:
            print(f"    {i+1}/{len(qa_items)} done")

    await client.close()
    return results


# ---------------------------------------------------------------------------
# Phase 3: Judge
# ---------------------------------------------------------------------------
async def judge_answers(results: List[Dict], judge_key: str) -> List[Dict]:
    """Grade using LLM-as-judge. Uses the larger model."""
    cfg = MODELS[judge_key]
    client = openai_lib.AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])
    correct = 0

    for i, r in enumerate(results):
        if r["response"].startswith("[ERROR]"):
            r["grade"] = "ERROR"
            continue

        prompt = f"""Grade if the generated answer matches the gold answer. Be generous — same topic/fact = CORRECT.

Question: {r['question']}
Gold answer: {r['gold_answer']}
Generated answer: {r['response']}

Respond JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "brief"}}"""

        try:
            resp = await client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": "You are an expert grader. JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            content = resp.choices[0].message.content or ""
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                j = json.loads(content[start:end + 1])
                r["grade"] = "CORRECT" if j.get("is_correct", "").upper() == "CORRECT" else "WRONG"
                r["reasoning"] = j.get("reasoning", "")
            else:
                r["grade"] = "PARSE_ERROR"
        except Exception as e:
            r["grade"] = "JUDGE_ERROR"
            r["reasoning"] = str(e)

        if r.get("grade") == "CORRECT":
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"    Judged {i+1}/{len(results)}: {correct}/{i+1} correct")

    await client.close()
    return results


# ---------------------------------------------------------------------------
# Metrics & Reporting
# ---------------------------------------------------------------------------
def compute_metrics(graded: List[Dict]) -> Dict[str, Any]:
    by_cat = {}
    total_c, total_n, total_t = 0, 0, 0.0
    for r in graded:
        cat = r.get("category_name", "unknown")
        if cat not in by_cat:
            by_cat[cat] = {"correct": 0, "total": 0}
        by_cat[cat]["total"] += 1
        total_n += 1
        total_t += r.get("time_s", 0)
        if r.get("grade") == "CORRECT":
            by_cat[cat]["correct"] += 1
            total_c += 1
    return {
        "accuracy": total_c / max(total_n, 1),
        "correct": total_c,
        "total": total_n,
        "avg_time_s": total_t / max(total_n, 1),
        "by_category": {
            cat: {"accuracy": d["correct"] / max(d["total"], 1), **d}
            for cat, d in sorted(by_cat.items())
        },
    }


def print_comparison(ma: Dict, la: str, mb: Dict, lb: str, inventory: Dict):
    print(f"\n{'=' * 90}")
    print(f"  LoCoMo MEMORY BENCHMARK — RESULTS")
    print(f"{'=' * 90}")

    # Memory inventory
    print(f"\n  EXTRACTED MEMORIES (shared context for both models):")
    total_mem = 0
    for cat in sorted(inventory.keys()):
        if cat == "episodes":
            continue
        items = inventory[cat]
        total_chars = sum(len(it["content"]) for it in items)
        print(f"    {cat}: {len(items)} files ({total_chars} chars)")
        total_mem += len(items)
    print(f"    TOTAL: {total_mem} structured memories (episodes excluded)")

    # QA comparison
    print(f"\n  {'METRIC':<25} | {la:>25} | {lb:>25}")
    print(f"  {'-' * 25}-+-{'-' * 25}-+-{'-' * 25}")
    print(f"  {'Overall accuracy':<25} | {ma['accuracy']:>24.1%} | {mb['accuracy']:>24.1%}")
    print(f"  {'Correct / Total':<25} | {ma['correct']:>15}/{ma['total']:<8} | {mb['correct']:>15}/{mb['total']:<8}")
    print(f"  {'Avg QA time':<25} | {ma['avg_time_s']:>23.2f}s | {mb['avg_time_s']:>23.2f}s")

    all_cats = set(ma["by_category"].keys()) | set(mb["by_category"].keys())
    if all_cats:
        print(f"\n  {'CATEGORY':<25} | {la:>25} | {lb:>25}")
        print(f"  {'-' * 25}-+-{'-' * 25}-+-{'-' * 25}")
        for cat in sorted(all_cats):
            a = ma["by_category"].get(cat, {"accuracy": 0, "correct": 0, "total": 0})
            b = mb["by_category"].get(cat, {"accuracy": 0, "correct": 0, "total": 0})
            print(f"  {cat:<25} | {a['accuracy']:>17.1%} ({a['correct']}/{a['total']}) | {b['accuracy']:>17.1%} ({b['correct']}/{b['total']})")

    # Winner
    if ma["accuracy"] != mb["accuracy"]:
        winner = la if ma["accuracy"] > mb["accuracy"] else lb
        wa, wb = (ma, mb) if ma["accuracy"] > mb["accuracy"] else (mb, ma)
        print(f"\n  Accuracy winner: {winner} ({wa['accuracy']:.1%} vs {wb['accuracy']:.1%})")
    else:
        print(f"\n  Accuracy: tie ({ma['accuracy']:.1%})")

    if ma["avg_time_s"] > 0 and mb["avg_time_s"] > 0:
        if ma["avg_time_s"] < mb["avg_time_s"]:
            print(f"  Speed winner: {la} ({mb['avg_time_s'] / ma['avg_time_s']:.1f}x faster)")
        else:
            print(f"  Speed winner: {lb} ({ma['avg_time_s'] / mb['avg_time_s']:.1f}x faster)")

    print(f"\n{'=' * 90}\n")


def save_csv(graded: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["question", "gold_answer", "response", "grade", "reasoning",
              "category", "category_name", "time_s", "tokens"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(graded)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
async def cleanup():
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
# CLI commands
# ---------------------------------------------------------------------------
def parse_range(s: str) -> Tuple[int, int]:
    if "-" in s:
        lo, hi = s.split("-", 1)
        return int(lo), int(hi)
    n = int(s)
    return n, n


async def cmd_import(args):
    samples = load_locomo(args.locomo, args.sample)
    session_range = parse_range(args.sessions) if args.sessions else None

    if args.clean:
        print("  Cleaning previous bench data...")
        await cleanup()

    for sample in samples:
        sid = sample["sample_id"]
        sessions = get_sessions(sample, session_range)
        print(f"\n  Sample {sid}: {len(sessions)} sessions ({sessions[0]['speakers'] if sessions else '?'})")
        await import_sessions(sessions, str(sid))


async def cmd_eval(args):
    samples = load_locomo(args.locomo, args.sample)

    # Wait for extraction
    await wait_for_extraction(args.wait)

    # Check what was extracted
    inventory = await get_memory_inventory()
    print(f"\n  Memory inventory:")
    total_mem = 0
    for cat in sorted(inventory.keys()):
        n = len(inventory[cat])
        print(f"    {cat}: {n}")
        if cat != "episodes":
            total_mem += n
    print(f"    Total (non-episode): {total_mem}")

    # Build context from structured memories only
    memory_context = await fetch_memory_context()
    print(f"  Memory context: {len(memory_context)} chars (~{len(memory_context) // 4} tokens)")

    if not memory_context.strip():
        print("  ERROR: No structured memories (entities/events/preferences) found.")
        print("  The server may need more time, or LoCoMo data didn't produce extractable memories.")
        print("  Check server logs for extraction errors.")
        return

    # QA items
    qa_items = []
    for sample in samples:
        qa_items.extend(get_qa_items(sample, args.count))
    print(f"  QA items: {len(qa_items)}")

    all_metrics = {}
    for model_key in MODELS:
        cfg = MODELS[model_key]
        print(f"\n  --- {cfg['label']} ---")
        results = await eval_qa(qa_items, model_key, memory_context)

        judge_key = "qwen"  # Always use larger model as judge
        print(f"  Judging with {MODELS[judge_key]['label']}...")
        graded = await judge_answers(results, judge_key)

        metrics = compute_metrics(graded)
        all_metrics[model_key] = metrics

        print(f"  {cfg['label']}: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']}) avg={metrics['avg_time_s']:.2f}s")

        save_csv(graded, RESULT_DIR / f"eval_{model_key}.csv")

    if len(all_metrics) == 2:
        keys = list(MODELS.keys())
        print_comparison(
            all_metrics[keys[0]], MODELS[keys[0]]["label"],
            all_metrics[keys[1]], MODELS[keys[1]]["label"],
            inventory,
        )


async def cmd_full(args):
    print("=" * 90)
    print("  LoCoMo MEMORY BENCHMARK")
    print("  Server extracts memories | Both models answer QA | Same context")
    print("=" * 90)

    # Import
    args.clean = True
    await cmd_import(args)

    # Eval (includes wait)
    await cmd_eval(args)


async def cmd_cleanup(args):
    await cleanup()
    print("  Done.")


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Memory Benchmark")
    parser.add_argument("--locomo", default=LOCOMO_PATH, help="Path to locomo10.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--sessions", default=None)
    p.add_argument("--clean", action="store_true")

    p = sub.add_parser("eval")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--wait", type=int, default=60, help="Seconds to wait for extraction")

    p = sub.add_parser("full")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--sessions", default="1-5")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--wait", type=int, default=60)

    p = sub.add_parser("cleanup")

    args = parser.parse_args()
    asyncio.run({"import": cmd_import, "eval": cmd_eval, "full": cmd_full, "cleanup": cmd_cleanup}[args.command](args))


if __name__ == "__main__":
    main()
