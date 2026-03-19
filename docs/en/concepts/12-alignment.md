# Pre-Response Alignment Check

A lightweight evaluation layer that checks agent responses against safety constraints and agent-specific rules. Runs entirely in the plugin layer — no OpenViking server changes required.

## Problem

- No pre-response evaluation exists in OpenViking
- Agents rely entirely on system prompts for alignment
- No mechanism to enforce constraints from stored memories or instructions
- No drift detection when alignment degrades over time

## Architecture

The alignment check runs **post-delivery** in the plugin's `afterTurn` hook. It cannot block the current response, but can inject corrections on the next turn.

```
Response delivered to user
         ↓
afterTurn (context-engine.ts)
  1. Extract last assistant text
  2. Assemble alignment profile (constraints)
  3. Layer 1: Regex pattern matching (zero latency)
  4. Layer 2: Keyword proximity check (zero latency)
  5. Layer 3: LLM evaluation (stub for v1)
  6. Feed drift detector
  7. Log verdict
  8. If flagged → store in pendingAlignmentFlags
         ↓
Next user message arrives
         ↓
before_prompt_build (index.ts)
  1. Check pendingAlignmentFlags
  2. If full_enforce mode → inject <alignment-correction>
  3. Clean stale flags (TTL: 30 min)
```

## Three Evaluation Layers

| Layer | Method | Latency | Catches |
|-------|--------|---------|---------|
| **L1** | Regex patterns | ~0ms | Credential leaks, PII, explicit violations |
| **L2** | Keyword proximity | ~0ms | Contextual constraint violations |
| **L3** | LLM evaluation | ~500ms | Subtle issues (stub in v1) |

## Default Safety Constraints

Three constraints are **always active**, even without agent-specific instructions:

| ID | Type | Description |
|----|------|-------------|
| `no_credential_leak` | hard_block | Detects exposed API keys, passwords, tokens |
| `no_pii_exposure` | soft_flag | Detects SSN patterns, credit card numbers |
| `no_harmful_instructions` | hard_block | Detects weapon/malware creation instructions |

## Agent-Specific Constraints

When agent instructions exist at `viking://agent/instructions/`, the alignment system extracts constraints using heuristic pattern matching:

- Imperative rules ("never", "must not", "do not") → `hard_block`
- Preference rules ("avoid", "prefer not", "should not") → `soft_flag`
- Keywords extracted for Layer 2 matching

## Three Modes

| Mode | hard_block | soft_flag | Correction |
|------|-----------|-----------|------------|
| **observe_only** | Log only | Log only | No |
| **soft_enforce** | Log + flag | Log only | No |
| **full_enforce** | Log + flag | Log + flag | Yes (next turn) |

**Rollout sequence**: Deploy with `observe_only` → review logs for 1-2 weeks → switch to `soft_enforce` → switch to `full_enforce` if quality is good.

## Drift Detection

A sliding window tracker monitors alignment scores over time:

```
DriftDetector:
  window: last N scores (default 20)
  alert when:
    - mean(window) < threshold (default 0.65)
    - consecutiveFlags >= limit (default 5)
```

Drift state is persisted to `~/.openclaw/memory/openviking/alignment/drift_state.json`.

## Configuration

Plugin config in `openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "openviking": {
        "config": {
          "alignment": {
            "enabled": true,
            "mode": "observe_only",
            "driftWindowSize": 20,
            "driftAlertThreshold": 0.65,
            "driftConsecutiveFlagLimit": 5
          }
        }
      }
    }
  }
}
```

## Monitoring

```bash
# Real-time alignment verdicts
docker compose logs openclaw-gateway -f 2>&1 | grep alignment

# Example output (pass):
# openviking: alignment verdict=pass score=1.00 constraints=3 mode=observe_only
#   drift=[evaluated=5,flagged=0,consecutive=0]

# Example output (violation):
# openviking: alignment verdict=hard_block score=0.00 constraints=3 mode=observe_only
# openviking: alignment issues: [L1:hard_block] Never expose API keys (matched: sk-abc123...)

# Drift state
cat ~/.openclaw/memory/openviking/alignment/drift_state.json
```

## Implementation

| Component | File |
|-----------|------|
| Alignment types + checks | `examples/openclaw-plugin/alignment.ts` |
| Drift detector | `examples/openclaw-plugin/drift.ts` |
| afterTurn evaluation | `examples/openclaw-plugin/context-engine.ts` |
| Correction injection | `examples/openclaw-plugin/index.ts` |
| Config schema | `examples/openclaw-plugin/config.ts`, `openclaw.plugin.json` |
| Helper | `examples/openclaw-plugin/text-utils.ts` (`extractLastAssistantText`) |

## Limitations

- Cannot block current response — correction happens on next turn
- Layer 3 (LLM evaluation) is a stub in v1
- Agent instruction extraction is heuristic-based (no LLM parsing in v1)
- True pre-delivery blocking would require an upstream OpenClaw hook

## Related Documents

- [Session Management](./08-session.md) — Session lifecycle
- [Distillation Pipeline](./11-distillation.md) — Memory evolution companion
- [Episodic Memory](./10-episodic-memory.md) — Conversation recall companion
