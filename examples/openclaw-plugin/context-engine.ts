import type { OpenVikingClient } from "./client.js";
import type { MemoryOpenVikingConfig } from "./config.js";
import type { AlignmentResult } from "./alignment.js";
import { alignmentCheck, assembleProfile } from "./alignment.js";
import { DriftDetector } from "./drift.js";
import { DEFAULT_MEMORY_OPENVIKING_DATA_DIR } from "./config.js";
import {
  getCaptureDecision,
  extractNewTurnTexts,
  extractNewTurnMessages,
  extractLastAssistantText,
  buildSkillUri,
} from "./text-utils.js";
import { createHash } from "crypto";
import {
  trimForLog,
  toJsonLog,
  summarizeExtractedMemories,
} from "./memory-ranking.js";

type AgentMessage = {
  role?: string;
  content?: unknown;
};

type ContextEngineInfo = {
  id: string;
  name: string;
  version?: string;
};

type AssembleResult = {
  messages: AgentMessage[];
  estimatedTokens: number;
  systemPromptAddition?: string;
};

type IngestResult = {
  ingested: boolean;
};

type IngestBatchResult = {
  ingestedCount: number;
};

type CompactResult = {
  ok: boolean;
  compacted: boolean;
  reason?: string;
  result?: unknown;
};

type ContextEngine = {
  info: ContextEngineInfo;
  ingest: (params: { sessionId: string; message: AgentMessage; isHeartbeat?: boolean }) => Promise<IngestResult>;
  ingestBatch?: (params: {
    sessionId: string;
    messages: AgentMessage[];
    isHeartbeat?: boolean;
  }) => Promise<IngestBatchResult>;
  afterTurn?: (params: {
    sessionId: string;
    sessionFile: string;
    messages: AgentMessage[];
    prePromptMessageCount: number;
    autoCompactionSummary?: string;
    isHeartbeat?: boolean;
    tokenBudget?: number;
    runtimeContext?: Record<string, unknown>;
  }) => Promise<void>;
  assemble: (params: { sessionId: string; messages: AgentMessage[]; tokenBudget?: number }) => Promise<AssembleResult>;
  compact: (params: {
    sessionId: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
    currentTokenCount?: number;
    compactionTarget?: "budget" | "threshold";
    customInstructions?: string;
    runtimeContext?: Record<string, unknown>;
  }) => Promise<CompactResult>;
};

type Logger = {
  info: (msg: string) => void;
  warn?: (msg: string) => void;
  error: (msg: string) => void;
};

function estimateTokens(messages: AgentMessage[]): number {
  return Math.max(1, messages.length * 80);
}

async function tryLegacyCompact(params: {
  sessionId: string;
  sessionFile: string;
  tokenBudget?: number;
  force?: boolean;
  currentTokenCount?: number;
  compactionTarget?: "budget" | "threshold";
  customInstructions?: string;
  runtimeContext?: Record<string, unknown>;
}): Promise<CompactResult | null> {
  const candidates = [
    "openclaw/context-engine/legacy",
    "openclaw/dist/context-engine/legacy.js",
  ];

  for (const path of candidates) {
    try {
      const mod = (await import(path)) as {
        LegacyContextEngine?: new () => {
          compact: (arg: typeof params) => Promise<CompactResult>;
        };
      };
      if (!mod?.LegacyContextEngine) {
        continue;
      }
      const legacy = new mod.LegacyContextEngine();
      return legacy.compact(params);
    } catch {
      // continue
    }
  }

  return null;
}

function warnOrInfo(logger: Logger, message: string): void {
  if (typeof logger.warn === "function") {
    logger.warn(message);
    return;
  }
  logger.info(message);
}

// Per-session capture throttle to prevent flooding OpenViking with rapid
// extract calls. Only successful captures (or captures that reach the server
// but extract 0 memories) trigger cooldown. Connection errors do NOT trigger
// cooldown so the next turn retries immediately.
const CAPTURE_COOLDOWN_MS = 60_000; // 60 seconds
const MAX_CONSECUTIVE_FAILURES = 3;
const _sessionLastCapture: Map<string, number> = new Map();
let _consecutiveCaptureFailures = 0;

// Instructions sync: only write when the system prompt hash changes
const _instructionsSyncedHash: Map<string, string> = new Map();

export function createMemoryOpenVikingContextEngine(params: {
  id: string;
  name: string;
  version?: string;
  cfg: Required<MemoryOpenVikingConfig>;
  logger: Logger;
  getClient: () => Promise<OpenVikingClient>;
  resolveAgentId: (sessionId: string) => string;
  pendingAlignmentFlags?: Map<string, AlignmentResult>;
}): ContextEngine {
  const {
    id,
    name,
    version,
    cfg,
    logger,
    getClient,
    resolveAgentId,
  } = params;

  const switchClientAgent = async (sessionId: string, phase: "assemble" | "afterTurn") => {
    const client = await getClient();
    const resolvedAgentId = resolveAgentId(sessionId);
    const before = client.getAgentId();
    if (resolvedAgentId && resolvedAgentId !== before) {
      client.setAgentId(resolvedAgentId);
      logger.info(`openviking: switched to agentId=${resolvedAgentId} for ${phase}`);
    }
    return client;
  };

  const driftDetector = cfg.alignment?.enabled
    ? new DriftDetector({
        windowSize: cfg.alignment.driftWindowSize,
        alertThreshold: cfg.alignment.driftAlertThreshold,
        consecutiveFlagLimit: cfg.alignment.driftConsecutiveFlagLimit,
        dataDir: DEFAULT_MEMORY_OPENVIKING_DATA_DIR,
        logger,
      })
    : null;

  return {
    info: {
      id,
      name,
      version,
    },

    async ingest(): Promise<IngestResult> {
      // Keep canonical capture behavior in afterTurn (same semantics as old agent_end hook).
      return { ingested: false };
    },

    async ingestBatch(): Promise<IngestBatchResult> {
      // Keep canonical capture behavior in afterTurn (same semantics as old agent_end hook).
      return { ingestedCount: 0 };
    },

    async assemble(assembleParams): Promise<AssembleResult> {
      return {
        messages: assembleParams.messages,
        estimatedTokens: estimateTokens(assembleParams.messages),
      };
    },

    async afterTurn(afterTurnParams): Promise<void> {
      if (!cfg.autoCapture) {
        return;
      }

      // Per-session cooldown: skip if this session captured recently
      const now = Date.now();
      const sessionKey = afterTurnParams.sessionId ?? "__global__";
      const lastCapture = _sessionLastCapture.get(sessionKey) ?? 0;
      if (now - lastCapture < CAPTURE_COOLDOWN_MS) {
        logger.info(
          `openviking: auto-capture skipped (cooldown, ${Math.round((CAPTURE_COOLDOWN_MS - (now - lastCapture)) / 1000)}s remaining)`,
        );
        return;
      }

      // Exponential backoff after consecutive failures
      if (_consecutiveCaptureFailures >= MAX_CONSECUTIVE_FAILURES) {
        const backoffMs = CAPTURE_COOLDOWN_MS * Math.pow(2, _consecutiveCaptureFailures - MAX_CONSECUTIVE_FAILURES);
        if (now - lastCapture < backoffMs) {
          logger.info(
            `openviking: auto-capture skipped (backoff after ${_consecutiveCaptureFailures} failures, ${Math.round((backoffMs - (now - lastCapture)) / 1000)}s remaining)`,
          );
          return;
        }
      }

      try {
        await switchClientAgent(afterTurnParams.sessionId, "afterTurn");

        // Sync agent instructions (system prompt) to OpenViking for alignment check
        const messages = afterTurnParams.messages ?? [];
        try {
          const systemMsg = messages.find(
            (m) => (m as Record<string, unknown>).role === "system",
          ) as Record<string, unknown> | undefined;
          const sysContent = typeof systemMsg?.content === "string" ? systemMsg.content : "";
          if (sysContent.length > 20) {
            const hash = createHash("md5").update(sysContent).digest("hex").slice(0, 16);
            const cached = _instructionsSyncedHash.get(sessionKey);
            if (cached !== hash) {
              const syncClient = await getClient();
              await syncClient.writeFile(
                "viking://agent/instructions/system_prompt.md",
                sysContent,
              );
              _instructionsSyncedHash.set(sessionKey, hash);
              logger.info(`openviking: synced agent instructions (hash=${hash}, len=${sysContent.length})`);
              // Prune old entries
              if (_instructionsSyncedHash.size > 200) {
                const keys = [..._instructionsSyncedHash.keys()];
                for (const k of keys.slice(0, keys.length - 50)) {
                  _instructionsSyncedHash.delete(k);
                }
              }
            }
          }
        } catch (err) {
          logger.info(`openviking: instructions sync skipped: ${String(err)}`);
        }
        if (messages.length === 0) {
          logger.info("openviking: auto-capture skipped (messages=0)");
          return;
        }

        const start =
          typeof afterTurnParams.prePromptMessageCount === "number" &&
          afterTurnParams.prePromptMessageCount >= 0
            ? afterTurnParams.prePromptMessageCount
            : 0;

        const { texts: newTexts, newCount } = extractNewTurnTexts(messages, start);

        if (newTexts.length === 0) {
          logger.info("openviking: auto-capture skipped (no new user/assistant messages)");
          return;
        }

        const turnText = newTexts.join("\n");
        const decision = getCaptureDecision(turnText, cfg.captureMode, cfg.captureMaxLength);
        const preview = turnText.length > 80 ? `${turnText.slice(0, 80)}...` : turnText;
        logger.info(
          "openviking: capture-check " +
            `shouldCapture=${String(decision.shouldCapture)} ` +
            `reason=${decision.reason} newMsgCount=${newCount} text=\"${preview}\"`,
        );

        if (!decision.shouldCapture) {
          logger.info("openviking: auto-capture skipped (capture decision rejected)");
          return;
        }

        // Use structured turn messages for richer session ingestion (includes tool calls)
        const { turns } = extractNewTurnMessages(messages, start);

        const client = await getClient();
        const sessionId = await client.createSession();
        try {
          // Ingest structured turns: send tool calls as structured ToolPart for skill extraction
          if (turns.length > 0) {
            for (const turn of turns) {
              if (turn.toolCalls?.length) {
                // Build structured parts: text content + tool parts
                const parts: Array<Record<string, unknown>> = [];
                if (turn.content.trim()) {
                  parts.push({ type: "text", text: turn.content.trim() });
                }
                for (const tc of turn.toolCalls) {
                  let toolInput: unknown;
                  try { toolInput = JSON.parse(tc.input); } catch { toolInput = { raw: tc.input }; }
                  parts.push({
                    type: "tool",
                    tool_name: tc.name,
                    tool_input: toolInput,
                    tool_output: (tc.result ?? "").slice(0, 1000),
                    tool_status: tc.result !== undefined ? "completed" : "pending",
                    skill_uri: buildSkillUri(tc.name),
                  });
                }
                if (parts.length > 0) {
                  await client.addSessionMessage(sessionId, turn.role, "", parts);
                }
              } else if (turn.content.trim()) {
                await client.addSessionMessage(sessionId, turn.role, turn.content.trim());
              }
            }
          } else {
            // Fallback to flat text if structured extraction yielded nothing
            await client.addSessionMessage(sessionId, "user", decision.normalizedText);
          }
          await client.getSession(sessionId).catch(() => ({}));
          const extracted = await client.extractSessionMemories(sessionId);

          logger.info(
            `openviking: auto-captured ${newCount} new messages, extracted ${extracted.length} memories`,
          );
          logger.info(
            `openviking: capture-detail ${toJsonLog({
              capturedCount: newCount,
              captured: [trimForLog(turnText, 260)],
              extractedCount: extracted.length,
              extracted: summarizeExtractedMemories(extracted),
            })}`,
          );
          if (extracted.length === 0) {
            warnOrInfo(
              logger,
              "openviking: auto-capture completed but extract returned 0 memories. " +
                "Check OpenViking server logs for embedding/extract errors.",
            );
          }
          // Capture reached the server — set cooldown and reset failures
          _sessionLastCapture.set(sessionKey, Date.now());
          _consecutiveCaptureFailures = 0;

          // Prune old session entries to prevent memory leak
          if (_sessionLastCapture.size > 200) {
            const cutoff = Date.now() - CAPTURE_COOLDOWN_MS * 10;
            for (const [k, t] of _sessionLastCapture) {
              if (t < cutoff) _sessionLastCapture.delete(k);
            }
          }
        } finally {
          await client.deleteSession(sessionId).catch(() => {});
        }
      } catch (err) {
        // Connection/server errors: do NOT set cooldown so next turn retries
        _consecutiveCaptureFailures++;
        warnOrInfo(
          logger,
          `openviking: auto-capture failed (${_consecutiveCaptureFailures}/${MAX_CONSECUTIVE_FAILURES}): ${String(err)}`,
        );
      }

      // P2: Alignment evaluation (post-delivery)
      if (cfg.alignment?.enabled && driftDetector) {
        try {
          const assistantText = extractLastAssistantText(afterTurnParams.messages ?? []);
          if (assistantText && assistantText.length > 20) {
            const client = await getClient();
            let instructionsText = "";
            try {
              instructionsText = await client.read("viking://agent/instructions/");
            } catch {
              // No instructions — only default constraints active
            }

            const profile = assembleProfile(instructionsText);
            const result = alignmentCheck(assistantText, profile);

            const alert = driftDetector.record(result);
            const driftState = driftDetector.getState();

            // Always log in observe_only for visibility
            const responsePreview = assistantText.length > 80
              ? `${assistantText.slice(0, 80)}...`
              : assistantText;
            logger.info(
              `openviking: alignment verdict=${result.verdict} score=${result.score.toFixed(2)} ` +
              `constraints=${profile.constraints.length} mode=${cfg.alignment.mode} ` +
              `drift=[evaluated=${driftState.totalEvaluated},flagged=${driftState.totalFlagged},consecutive=${driftState.consecutiveFlags}] ` +
              `response="${responsePreview}"`,
            );

            if (result.verdict !== "pass") {
              logger.info(
                `openviking: alignment issues: ${result.issues.map((i) => `[L${i.layer}:${i.type}] ${i.description}${i.matchedText ? ` (matched: ${i.matchedText})` : ""}`).join("; ")}`,
              );
              if (params.pendingAlignmentFlags) {
                params.pendingAlignmentFlags.set(afterTurnParams.sessionId, result);
              }
            }
            if (alert) {
              warnOrInfo(
                logger,
                `openviking: DRIFT ALERT — mean=${alert.mean.toFixed(2)}, consecutiveFlags=${alert.consecutiveFlags}, totalEvaluated=${alert.totalEvaluated}`,
              );
            }
          }
        } catch (err) {
          warnOrInfo(logger, `openviking: alignment check failed: ${String(err)}`);
        }
      }
    },

    async compact(compactParams): Promise<CompactResult> {
      const delegated = await tryLegacyCompact(compactParams);
      if (delegated) {
        return delegated;
      }

      warnOrInfo(
        logger,
        "openviking: legacy compaction delegation unavailable; skipping compact",
      );

      return {
        ok: true,
        compacted: false,
        reason: "legacy_compact_unavailable",
      };
    },
  };
}
