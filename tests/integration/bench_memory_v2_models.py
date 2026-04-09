#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Benchmark: Memory Extraction V2 — Qwen3.5-35B-A3B vs Bonsai-8B

Compares result quality and speed for memory management LLM use case.
Calls each model endpoint directly via OpenAI-compatible API.

Usage:
    cd source/
    source .venv/bin/activate
    python tests/integration/bench_memory_v2_models.py
    python tests/integration/bench_memory_v2_models.py --runs 3   # consistency test
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import openai
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODELS = {
    "qwen3.5-35b-a3b": {
        "api_base": "http://localhost:1234/v1",
        "api_key": "not-needed",
        "model": "Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-Q4_K_M.gguf",
        "label": "Qwen3.5-35B-A3B (port 1234)",
    },
    "bonsai-8b": {
        "api_base": "http://localhost:8080/v1",
        "api_key": "not-needed",
        "model": "Bonsai-8B.gguf",
        "label": "Bonsai-8B (port 8080)",
    },
}

# ---------------------------------------------------------------------------
# Test conversations
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# LONG conversation — covers all 6 target memory types with clear signals
# Expected: preferences(2+), tools(1+), cases(2+), patterns(1+), events(2+), entities(3+)
# ---------------------------------------------------------------------------
CONVERSATION_LONG = [
    # --- Entity signals: people, projects, systems ---
    ("user", "I'm setting up the staging environment for Kipa Cuan. That's our new API gateway — built on Envoy, managed by the infra team led by Sarah Wong."),
    ("assistant", "Got it. What's the current state of the staging setup?"),

    # --- Case signal: problem → solution ---
    ("user", "We hit a wall yesterday. Redis on staging got OOM killed. Turned out maxmemory-policy was set to noeviction with only 2GB limit. Session keys had no TTL so memory just filled up."),
    ("assistant", "How did you resolve it?"),
    ("user", "I did a cluster failover to the replica first, then changed the policy to allkeys-lru, added 24h TTL on all session keys, and bumped memory to 4GB. Everything came back up and we resumed load testing."),
    ("assistant", "Good recovery. Any other issues?"),

    # --- Second case signal ---
    ("user", "Yeah, after fixing Redis I found port 5432 was conflicting — the host PostgreSQL was still running and blocking our containerized Postgres. Had to stop the host service first with systemctl stop postgresql, then the container came up fine."),
    ("assistant", "Classic port conflict. How's the overall deployment workflow looking?"),

    # --- Pattern signal: reusable workflow ---
    ("user", "For Kipa Cuan deployments we now follow a strict sequence: first run health checks on the k8s cluster, then do a rolling update with maxSurge=1 and maxUnavailable=0, wait for all pods healthy, run smoke tests, and only then switch the load balancer. If smoke tests fail, we rollback immediately."),
    ("assistant", "That's a solid deployment pattern. Any preferences on the tooling side?"),

    # --- Preference signals ---
    ("user", "Yes — I strongly prefer using Helm charts over raw kubectl manifests. Much easier to version and rollback. And for monitoring, I always want Grafana dashboards with at least p50, p95, p99 latency panels — I don't trust averages alone."),
    ("assistant", "Makes sense. What about your development environment preferences?"),
    ("user", "I use VS Code with the Remote SSH extension for all staging work. I refuse to edit production configs through a web UI — always SSH + vim for that. And I need dark mode everywhere, light themes give me headaches."),
    ("assistant", "Noted. Any upcoming events or deadlines I should know about?"),

    # --- Event signals: dates, plans ---
    ("user", "The Kipa Cuan load test is scheduled for April 12th. We need to hit 10k RPS sustained for 30 minutes. If it passes, the production rollout is planned for April 20th. Sarah is presenting the results to the CTO on April 22nd."),
    ("assistant", "Tight timeline. How are you tracking all this?"),

    # --- Tool signal: specific tool usage with outcome ---
    ("user", "I've been using k9s for cluster monitoring during the tests — it's way better than kubectl for real-time pod watching. I ran it about 15 times during yesterday's debugging session, and it helped me spot the OOM issue within seconds. The only downside is it sometimes crashes when you have 200+ pods, so I keep kubectl as backup."),
    ("assistant", "Good to know about k9s. Anything else?"),

    # --- Another event + entity ---
    ("user", "One more thing — our new team member, David Li, starts on April 8th. He'll be taking over the Redis cluster management from me. I need to write a handover doc before then."),
]

# ---------------------------------------------------------------------------
# SHORT conversation — still covers all 6 types, just compressed
# Expected: preferences(1), tools(1), cases(1), patterns(1), events(1), entities(1+)
# ---------------------------------------------------------------------------
CONVERSATION_SHORT = [
    ("user", "Quick update: I fixed the SSL cert issue on api.kipa-cuan.dev today. The cert had expired because certbot's cron was misconfigured. I renewed it with certbot renew, restarted nginx, and set up a proper monthly cron. Going forward, always check cert expiry with openssl s_client before attempting renewal."),
    ("assistant", "Good fix. Anything else?"),
    ("user", "Yeah — the security audit with Jianguo Tech is confirmed for April 18th. I told them I prefer all reports in PDF format with CVSS scores, not just raw findings. Also, I've been using trivy for container scanning — it caught 3 critical CVEs in our base image yesterday, runs in about 40 seconds per scan."),
]

# Memory extraction prompt matching OpenViking v2 schema types
SYSTEM_PROMPT = """You are a memory extraction system for an AI agent's long-term memory store. Analyze the conversation and extract structured memories into these exact types:

## Memory Types

1. **preferences** — "what the user likes/dislikes or is accustomed to"
   Fields: {"memory_type": "preferences", "topic": "...", "content": "..."}
   topic: semantic description like "deployment_tooling", "monitoring_style", "editor_preference"

2. **entities** — named things (people, projects, systems) as Zettelkasten cards
   Fields: {"memory_type": "entities", "name": "...", "content": "..."}
   name: lowercase_underscores, max 3 words (e.g. "kipa_cuan", "sarah_wong")

3. **events** — time-bound activities with specific dates
   Fields: {"memory_type": "events", "event_name": "...", "summary": "...", "date": "..."}
   Always convert relative dates to absolute (today = 2026-04-06). Include commitments and plans.

4. **cases** — "problem → solution" pairs worth remembering
   Fields: {"memory_type": "cases", "case_name": "Problem → Solution", "problem": "...", "solution": "..."}

5. **patterns** — reusable workflows/processes ("when X, do Y")
   Fields: {"memory_type": "patterns", "pattern_name": "Process: Step description", "content": "..."}

6. **tools** — tool usage insights with success/failure observations
   Fields: {"memory_type": "tools", "tool_name": "...", "best_for": "...", "common_failures": "...", "recommendation": "..."}

## Rules
- Extract ONLY genuinely useful long-term information
- Do NOT extract greetings, filler, or trivially obvious facts
- Each memory should clearly belong to exactly one type
- Return ONLY a valid JSON array of memory objects, no other text"""


@dataclass
class BenchResult:
    model_key: str
    label: str
    run_id: int = 0
    # Timing
    extraction_time_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Quality
    memories_found: int = 0
    memory_categories: Dict[str, int] = field(default_factory=dict)
    memory_samples: List[str] = field(default_factory=list)
    raw_output: str = ""
    valid_json: bool = False
    error: Optional[str] = None


async def run_extraction(
    client: openai.AsyncOpenAI,
    model: str,
    conversation: List[Tuple[str, str]],
    model_key: str,
    label: str,
    run_id: int = 0,
) -> BenchResult:
    """Run a single memory extraction call."""
    r = BenchResult(model_key=model_key, label=label, run_id=run_id)

    conv_text = "\n".join(f"[{role}]: {content}" for role, content in conversation)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract memories from this conversation:\n\n{conv_text}"},
    ]

    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
        )
        r.extraction_time_s = time.perf_counter() - t0

        # Token usage
        if resp.usage:
            r.prompt_tokens = resp.usage.prompt_tokens or 0
            r.completion_tokens = resp.usage.completion_tokens or 0
            r.total_tokens = resp.usage.total_tokens or 0

        content = resp.choices[0].message.content or ""
        r.raw_output = content

        # Parse JSON
        clean = content.strip()
        # Strip markdown fences
        if clean.startswith("```"):
            first_nl = clean.find("\n")
            if first_nl != -1:
                clean = clean[first_nl + 1:]
        if clean.endswith("```"):
            clean = clean[:-3].rstrip()
        clean = clean.strip()

        try:
            memories = json.loads(clean)
            r.valid_json = True
            if isinstance(memories, dict):
                # Unwrap if wrapped
                for key in ("memories", "results", "data"):
                    if key in memories and isinstance(memories[key], list):
                        memories = memories[key]
                        break
                else:
                    memories = [memories]
            if isinstance(memories, list):
                r.memories_found = len(memories)
                for m in memories:
                    mt = m.get("memory_type", "unknown")
                    r.memory_categories[mt] = r.memory_categories.get(mt, 0) + 1
                    if len(r.memory_samples) < 10:
                        # Flexible title: different schemas use different name fields
                        title = (
                            m.get("title")
                            or m.get("name")
                            or m.get("case_name")
                            or m.get("pattern_name")
                            or m.get("event_name")
                            or m.get("tool_name")
                            or m.get("topic")
                            or "?"
                        )
                        conf = m.get("confidence", "")
                        snippet = (m.get("content") or m.get("summary") or m.get("solution") or m.get("best_for") or "")[:120]
                        conf_str = f"|conf={conf}" if conf else ""
                        r.memory_samples.append(f"  [{mt}{conf_str}] {title}\n    {snippet}")
        except (json.JSONDecodeError, TypeError) as e:
            r.valid_json = False
            r.memory_samples.append(f"  JSON parse error: {e}")
            r.memory_samples.append(f"  Raw (first 300): {content[:300]}")

    except Exception as e:
        r.extraction_time_s = time.perf_counter() - t0
        r.error = str(e)

    return r


def print_result(r: BenchResult, verbose: bool = True):
    if r.error:
        print(f"    ERROR: {r.error}")
        return
    print(f"    Time:       {r.extraction_time_s:>7.2f}s")
    print(f"    Tokens:     {r.prompt_tokens} prompt + {r.completion_tokens} completion = {r.total_tokens} total")
    print(f"    Valid JSON: {r.valid_json}")
    print(f"    Memories:   {r.memories_found}")
    if r.memory_categories:
        cats = ", ".join(f"{k}={v}" for k, v in sorted(r.memory_categories.items()))
        print(f"    Categories: {cats}")
    if verbose and r.memory_samples:
        print(f"    Samples:")
        for s in r.memory_samples:
            print(f"    {s}")


def print_comparison(results_a: List[BenchResult], results_b: List[BenchResult]):
    """Compare two sets of results."""
    valid_a = [r for r in results_a if not r.error]
    valid_b = [r for r in results_b if not r.error]

    if not valid_a or not valid_b:
        print("  Cannot compare — one or both models had no successful runs.")
        return

    # Averages
    avg_time_a = sum(r.extraction_time_s for r in valid_a) / len(valid_a)
    avg_time_b = sum(r.extraction_time_s for r in valid_b) / len(valid_b)
    avg_mem_a = sum(r.memories_found for r in valid_a) / len(valid_a)
    avg_mem_b = sum(r.memories_found for r in valid_b) / len(valid_b)
    avg_tok_a = sum(r.total_tokens for r in valid_a) / len(valid_a)
    avg_tok_b = sum(r.total_tokens for r in valid_b) / len(valid_b)

    label_a = valid_a[0].label
    label_b = valid_b[0].label

    print(f"\n  {'METRIC':<25} {'|':>1} {label_a:>30} {'|':>1} {label_b:>30}")
    print(f"  {'-'*25}-+-{'-'*30}-+-{'-'*30}")
    print(f"  {'Avg extraction time':<25} | {avg_time_a:>28.2f}s | {avg_time_b:>28.2f}s")
    print(f"  {'Avg memories extracted':<25} | {avg_mem_a:>29.1f} | {avg_mem_b:>29.1f}")
    print(f"  {'Avg total tokens':<25} | {avg_tok_a:>29.0f} | {avg_tok_b:>29.0f}")
    print(f"  {'Valid JSON rate':<25} | {sum(1 for r in valid_a if r.valid_json)/len(valid_a)*100:>28.0f}% | {sum(1 for r in valid_b if r.valid_json)/len(valid_b)*100:>28.0f}%")
    print(f"  {'Success rate':<25} | {len(valid_a)/len(results_a)*100:>28.0f}% | {len(valid_b)/len(results_b)*100:>28.0f}%")

    # Speed comparison
    if avg_time_a > 0 and avg_time_b > 0:
        if avg_time_a < avg_time_b:
            print(f"\n  Speed winner: {label_a} ({avg_time_b/avg_time_a:.1f}x faster)")
        else:
            print(f"\n  Speed winner: {label_b} ({avg_time_a/avg_time_b:.1f}x faster)")

    # Quality comparison
    if avg_mem_a > avg_mem_b:
        print(f"  Quality winner: {label_a} ({avg_mem_a:.1f} vs {avg_mem_b:.1f} memories)")
    elif avg_mem_b > avg_mem_a:
        print(f"  Quality winner: {label_b} ({avg_mem_b:.1f} vs {avg_mem_a:.1f} memories)")
    else:
        print(f"  Quality: tie ({avg_mem_a:.1f} memories each)")

    # Category coverage
    EXPECTED_TYPES = {"preferences", "entities", "events", "cases", "patterns", "tools"}
    cats_a = set()
    cats_b = set()
    for r in valid_a:
        cats_a |= set(r.memory_categories.keys())
    for r in valid_b:
        cats_b |= set(r.memory_categories.keys())
    only_a = cats_a - cats_b
    only_b = cats_b - cats_a
    shared = cats_a & cats_b
    print(f"\n  Category coverage (expected: {', '.join(sorted(EXPECTED_TYPES))}):")
    print(f"    Shared:            {', '.join(sorted(shared)) or 'none'}")
    if only_a:
        print(f"    Only {label_a}: {', '.join(sorted(only_a))}")
    if only_b:
        print(f"    Only {label_b}: {', '.join(sorted(only_b))}")
    missing_a = EXPECTED_TYPES - cats_a
    missing_b = EXPECTED_TYPES - cats_b
    if missing_a:
        print(f"    MISSING from {label_a}: {', '.join(sorted(missing_a))}")
    if missing_b:
        print(f"    MISSING from {label_b}: {', '.join(sorted(missing_b))}")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark memory extraction v2 models")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per model (default: 1)")
    parser.add_argument("--short", action="store_true", help="Also test short conversation")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    verbose = not args.quiet
    n_runs = args.runs

    print("=" * 90)
    print("  MEMORY EXTRACTION V2 BENCHMARK")
    print("  Qwen3.5-35B-A3B (port 1234) vs Bonsai-8B (port 8080)")
    print(f"  Runs per model: {n_runs}")
    print("=" * 90)

    # Verify endpoints are up
    for key, cfg in MODELS.items():
        try:
            c = openai.AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])
            models = await c.models.list()
            print(f"  {cfg['label']}: OK ({models.data[0].id if models.data else '?'})")
            await c.close()
        except Exception as e:
            print(f"  {cfg['label']}: UNREACHABLE — {e}")
            print("  Aborting.")
            return

    all_results: Dict[str, List[BenchResult]] = {k: [] for k in MODELS}

    # --- Rich conversation benchmark ---
    print(f"\n{'—' * 90}")
    print(f"  TEST: Rich Memory Management Conversation ({len(CONVERSATION_LONG)} messages)")
    print(f"{'—' * 90}")

    for model_key, cfg in MODELS.items():
        client = openai.AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])

        # Warmup
        try:
            await client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
        except Exception:
            pass

        for i in range(n_runs):
            print(f"\n  [{cfg['label']}] Run {i+1}/{n_runs}")
            r = await run_extraction(
                client, cfg["model"], CONVERSATION_LONG, model_key, cfg["label"], i + 1
            )
            all_results[model_key].append(r)
            print_result(r, verbose=verbose)

        await client.close()

    # Print comparison
    keys = list(MODELS.keys())
    print(f"\n{'=' * 90}")
    print(f"  COMPARISON — Rich Conversation")
    print(f"{'=' * 90}")
    print_comparison(all_results[keys[0]], all_results[keys[1]])

    # --- Short conversation (optional) ---
    if args.short:
        short_results: Dict[str, List[BenchResult]] = {k: [] for k in MODELS}

        print(f"\n{'—' * 90}")
        print(f"  TEST: Short Factual Conversation ({len(CONVERSATION_SHORT)} messages)")
        print(f"{'—' * 90}")

        for model_key, cfg in MODELS.items():
            client = openai.AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])
            for i in range(n_runs):
                print(f"\n  [{cfg['label']}] Run {i+1}/{n_runs}")
                r = await run_extraction(
                    client, cfg["model"], CONVERSATION_SHORT, model_key, cfg["label"], i + 1
                )
                short_results[model_key].append(r)
                print_result(r, verbose=verbose)
            await client.close()

        print(f"\n{'=' * 90}")
        print(f"  COMPARISON — Short Conversation")
        print(f"{'=' * 90}")
        print_comparison(short_results[keys[0]], short_results[keys[1]])

    # --- Consistency report (if multiple runs) ---
    if n_runs >= 2:
        print(f"\n{'=' * 90}")
        print(f"  CONSISTENCY REPORT ({n_runs} runs)")
        print(f"{'=' * 90}")

        for model_key in MODELS:
            runs = all_results[model_key]
            valid = [r for r in runs if not r.error]
            label = runs[0].label if runs else model_key
            if not valid:
                print(f"\n  {label}: ALL ERRORS")
                continue

            times = [r.extraction_time_s for r in valid]
            counts = [r.memories_found for r in valid]

            mean_t = sum(times) / len(times)
            std_t = (sum((t - mean_t) ** 2 for t in times) / max(len(times) - 1, 1)) ** 0.5

            print(f"\n  {label} ({len(valid)}/{len(runs)} success)")
            print(f"    Time:     avg={mean_t:.2f}s  std={std_t:.2f}s  min={min(times):.2f}s  max={max(times):.2f}s")
            print(f"    Memories: avg={sum(counts)/len(counts):.1f}  min={min(counts)}  max={max(counts)}")

            cats = [set(r.memory_categories.keys()) for r in valid]
            if len(cats) >= 2:
                common = cats[0]
                all_cats = set()
                for s in cats:
                    common &= s
                    all_cats |= s
                print(f"    Stable categories:   {', '.join(sorted(common)) or 'none'}")
                unstable = all_cats - common
                if unstable:
                    print(f"    Unstable categories: {', '.join(sorted(unstable))}")

    print(f"\n{'=' * 90}")
    print(f"  DONE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    asyncio.run(main())
