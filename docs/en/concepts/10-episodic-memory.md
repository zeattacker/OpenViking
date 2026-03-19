# Episodic Memory

Episodic memory captures structured conversation summaries that persist across sessions, enabling agents to recall "what we discussed" — not just extracted facts.

## Problem

Standard memory extraction (session commit) distills conversations into discrete facts (preferences, entities, events). The conversational context — flow of discussion, decisions made, unresolved items — is lost after extraction. Agents cannot answer temporal queries like "what did we discuss about the database migration?" or "who was assigned to the infrastructure work?"

## Solution

At session commit, alongside memory extraction, OpenViking generates an **episode summary** — a structured markdown document that captures the conversation as a narrative.

## Architecture

```
Session Commit
  ├── Archive messages (existing)
  ├── Generate L0/L1 for archive (existing)
  ├── Extract memories (existing)
  └── [NEW] Generate + index episode summary
```

### Episode Flow

```
Messages → LLM (episode_summary prompt) → Episode .md → Write to AGFS
                                                          ↓
                                                    Vectorize (L2)
                                                          ↓
                                                    Semantic Queue (L0/L1)
```

## Episode Document Format

```markdown
# Episode: {title}

## Summary
Narrative description of the conversation (2-4 sentences).

## Key Topics
- Topic 1
- Topic 2

## Decisions & Outcomes
- Decision: rationale

## Entities Mentioned
- Entity: role in this conversation

## Unresolved Items
- Item left open for future discussion
```

## Storage

```
viking://user/{user_space}/episodes/
  ep_{session_id}_{timestamp}.md    # Episode (L2)
  .abstract.md                       # L0 — auto-generated
  .overview.md                       # L1 — auto-generated
```

Episodes are stored **per-user** (not per-agent). All agents for the same user share the episodes directory, since episodes capture the user's conversation history regardless of which agent was involved.

## Dynamic Token Budget

The episode summary prompt uses a dynamic token budget based on conversation length:

| Messages | Max Tokens | Rationale |
|----------|------------|-----------|
| ≤ 5      | 300        | Short exchange — brief summary |
| ≤ 15     | 500        | Normal session |
| ≤ 40     | 800        | Long session — more topics |
| > 40     | 1200       | Marathon session — comprehensive |

## Retrieval

Episodes are vector-indexed at L2 and searchable via `find()`:

```python
results = await client.find(
    "what did we discuss about database migration",
    target_uri="viking://user/episodes"
)
```

The plugin provides a dedicated `memory_recall_episodes` tool with optional date filtering (`after_date`, `before_date`).

## Configuration

Episode generation is automatic when `auto_generate_l0` and `auto_generate_l1` are enabled in `ov.conf`. No additional configuration required.

## Implementation

| Component | File |
|-----------|------|
| EpisodeIndexer | `openviking/session/episode_indexer.py` |
| Prompt template | `openviking/prompts/templates/compression/episode_summary.yaml` |
| Compressor hook | `openviking/session/compressor.py` (after memory extraction) |
| Plugin tool | `examples/openclaw-plugin/index.ts` (`memory_recall_episodes`) |

## Related Documents

- [Session Management](./08-session.md) — Session lifecycle and memory extraction
- [Context Extraction](./06-extraction.md) — L0/L1/L2 generation pipeline
- [Retrieval Mechanism](./07-retrieval.md) — How episodes are searched
