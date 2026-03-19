# Distillation Pipeline

The distillation pipeline transforms OpenViking's one-shot memory extraction into a continuous knowledge evolution system where memories consolidate, evolve, and decay over time.

## Problem

Without distillation:
- 10 similar case memories remain as 10 separate entries (no consolidation)
- Memories are never enriched with confirming evidence
- Stale, irrelevant memories persist forever
- No signal for which memories are actively useful

## Three Capabilities

### P1a: Pattern Consolidation

Clusters similar case memories and consolidates them into reusable patterns.

```
Every 6 hours:
  1. Scan cases/ in each agent space
  2. Compute pairwise cosine similarity
  3. Union-find clustering (threshold ≥ 0.85)
  4. For clusters with ≥ 3 members:
     → LLM: generate consolidated pattern
     → Write to patterns/consolidated_{hash}.md
     → Vectorize + trigger L0/L1 regeneration
```

**Concurrency control**: File-based lock (`/tmp/openviking_distill_{scope_hash}.lock`) with 1-hour staleness detection. Session commits are never blocked — new memories are picked up in the next cycle.

### P1b: Memory Evolution (EVOLVE)

During memory extraction, when new evidence reinforces an existing memory without contradicting it, the EVOLVE action enriches the existing memory rather than creating a duplicate or merging.

```
Extraction pipeline:
  Candidate → Vector pre-filter → Find similar → LLM dedup decision
                                                       ↓
                                          skip | create | none
                                                   ↓
                                     merge | delete | EVOLVE (new)
```

| Action | Behavior |
|--------|----------|
| **merge** | Rewrite: combine candidate + existing into unified content |
| **evolve** | Enrich: preserve existing content, append new evidence |
| **delete** | Remove: candidate fully invalidates existing memory |

EVOLVE updates metadata:
- `evolution_count` incremented
- `last_confirmed` set to current timestamp

### P1c: Recall Tracking + Memory Decay

**Recall tracking**: Every time a memory is recalled (via tool or auto-recall), its `active_count` is incremented in the vector DB via the `/api/v1/search/track-recall` endpoint.

**Decay**: The `MemoryArchiver` periodically scans memories using `hotness_score()`:

```
hotness_score = sigmoid(log1p(active_count)) * exp_decay(age)
```

Memories below the threshold (default 0.1) and older than `min_age_days` (default 7) are moved to `_archive/`. Archived memories are excluded from retrieval but remain recoverable.

## Architecture

```
DistillationScheduler (asyncio loops)
  ├── Consolidation loop (every 6h)
  │   └── PatternDistiller.consolidate(scope)
  │       ├── Scan cases/ via viking_fs.ls()
  │       ├── Get vectors via vikingdb.scroll()
  │       ├── Cluster (union-find, cosine ≥ 0.85)
  │       └── LLM consolidate → write pattern
  │
  └── Decay loop (every 24h)
      └── MemoryArchiver.scan_and_archive(scope)
          ├── Scroll L2 memories
          ├── Compute hotness_score()
          └── Archive cold memories to _archive/
```

## Configuration

```json
{
  "distillation": {
    "enabled": true,
    "consolidation_enabled": true,
    "consolidation_interval_hours": 6,
    "decay_enabled": true,
    "decay_check_interval_hours": 24
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | false | Master switch |
| `consolidation_enabled` | true | Enable pattern consolidation |
| `consolidation_interval_hours` | 6 | Hours between consolidation runs |
| `consolidation_similarity_threshold` | 0.85 | Cosine similarity for clustering |
| `consolidation_min_cluster_size` | 3 | Minimum cluster size |
| `decay_enabled` | true | Enable memory decay/archival |
| `decay_check_interval_hours` | 24 | Hours between decay scans |

## Implementation

| Component | File |
|-----------|------|
| PatternDistiller | `openviking/session/distiller.py` |
| DistillationScheduler | `openviking/session/distillation_scheduler.py` |
| MemoryEvolver | `openviking/session/memory_evolver.py` |
| EVOLVE action | `openviking/session/memory_deduplicator.py` |
| EVOLVE handler | `openviking/session/compressor.py` |
| Consolidation prompt | `openviking/prompts/templates/compression/pattern_consolidation.yaml` |
| Evolution prompt | `openviking/prompts/templates/compression/memory_evolution.yaml` |
| Dedup prompt (v3.4.0) | `openviking/prompts/templates/compression/dedup_decision.yaml` |
| Recall tracking endpoint | `openviking/server/routers/search.py` |
| Plugin recall tracking | `examples/openclaw-plugin/index.ts`, `client.ts` |
| DistillationConfig | `openviking_cli/utils/config/open_viking_config.py` |
| Service wiring | `openviking/service/core.py` |

## Related Documents

- [Session Management](./08-session.md) — Memory extraction and dedup decisions
- [Context Extraction](./06-extraction.md) — L0/L1/L2 pipeline
- [Episodic Memory](./10-episodic-memory.md) — Companion feature for conversation recall
