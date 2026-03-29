import { describe, expect, it } from "vitest";

import type { AgentMessage } from "../tool-call-id.js";
import {
  repairToolCallInputs,
  repairToolUseResultPairing,
  stripToolResultDetails,
  makeMissingToolResult,
} from "../session-transcript-repair.js";

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function assistantWithToolCalls(
  calls: Array<{ id: string; name: string; input?: unknown }>,
  extras?: Partial<Extract<AgentMessage, { role: "assistant" }>>,
): Extract<AgentMessage, { role: "assistant" }> {
  return {
    role: "assistant",
    content: calls.map((c) => ({
      type: "toolCall" as const,
      id: c.id,
      name: c.name,
      input: c.input ?? {},
    })),
    ...extras,
  } as Extract<AgentMessage, { role: "assistant" }>;
}

function toolResult(
  toolCallId: string,
  toolName?: string,
  opts?: Partial<Extract<AgentMessage, { role: "toolResult" }>>,
): Extract<AgentMessage, { role: "toolResult" }> {
  return {
    role: "toolResult",
    toolCallId,
    toolName: toolName ?? "unknown",
    content: [{ type: "text", text: "result" }],
    isError: false,
    ...opts,
  } as Extract<AgentMessage, { role: "toolResult" }>;
}

// ---------------------------------------------------------------------------
// repairToolCallInputs
// ---------------------------------------------------------------------------

describe("repairToolCallInputs", () => {
  it("passes through valid messages unchanged", () => {
    const messages: AgentMessage[] = [
      { role: "user", content: "hello" },
      assistantWithToolCalls([{ id: "tc1", name: "bash", input: { cmd: "ls" } }]),
      toolResult("tc1", "bash"),
    ];
    const report = repairToolCallInputs(messages);
    expect(report.messages).toBe(messages);
    expect(report.droppedToolCalls).toBe(0);
    expect(report.droppedAssistantMessages).toBe(0);
  });

  it("drops tool call blocks missing id", () => {
    const messages: AgentMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "toolCall", id: "", name: "bash", input: { cmd: "ls" } },
          { type: "text", text: "thinking" },
        ],
      },
    ];
    const report = repairToolCallInputs(messages);
    expect(report.droppedToolCalls).toBe(1);
    expect((report.messages[0] as { content: unknown[] }).content).toHaveLength(1);
  });

  it("drops tool call blocks missing input", () => {
    const messages: AgentMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "toolCall", id: "tc1", name: "bash" },
        ],
      },
    ];
    const report = repairToolCallInputs(messages);
    expect(report.droppedToolCalls).toBe(1);
  });

  it("drops tool call blocks with disallowed name", () => {
    const messages: AgentMessage[] = [
      assistantWithToolCalls([{ id: "tc1", name: "evil_tool", input: {} }]),
    ];
    const report = repairToolCallInputs(messages, { allowedToolNames: ["bash", "read_file"] });
    expect(report.droppedToolCalls).toBe(1);
  });

  it("drops entire assistant message when all tool calls are dropped", () => {
    const messages: AgentMessage[] = [
      {
        role: "assistant",
        content: [
          { type: "toolCall", id: "", name: "bash", input: {} },
        ],
      },
    ];
    const report = repairToolCallInputs(messages);
    expect(report.droppedAssistantMessages).toBe(1);
    expect(report.messages).toHaveLength(0);
  });

  it("non-assistant messages pass through unchanged", () => {
    const user: AgentMessage = { role: "user", content: "hi" };
    const tr = toolResult("tc1", "bash");
    const report = repairToolCallInputs([user, tr]);
    expect(report.messages).toEqual([user, tr]);
    expect(report.droppedToolCalls).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// repairToolUseResultPairing
// ---------------------------------------------------------------------------

describe("repairToolUseResultPairing", () => {
  it("passes through properly paired messages unchanged", () => {
    const messages: AgentMessage[] = [
      { role: "user", content: "do something" },
      assistantWithToolCalls([{ id: "tc1", name: "bash", input: {} }]),
      toolResult("tc1", "bash"),
    ];
    const report = repairToolUseResultPairing(messages);
    expect(report.messages).toBe(messages);
    expect(report.added).toHaveLength(0);
    expect(report.droppedDuplicateCount).toBe(0);
    expect(report.droppedOrphanCount).toBe(0);
    expect(report.moved).toBe(false);
  });

  it("inserts synthetic error toolResult for missing results", () => {
    const messages: AgentMessage[] = [
      assistantWithToolCalls([{ id: "tc1", name: "bash", input: {} }]),
      // no toolResult for tc1
      { role: "user", content: "next" },
    ];
    const report = repairToolUseResultPairing(messages);
    expect(report.added).toHaveLength(1);
    expect(report.added[0].toolCallId).toBe("tc1");
    expect(report.added[0].isError).toBe(true);
    // Synthetic result should be right after assistant
    expect((report.messages[1] as { role: string }).role).toBe("toolResult");
    expect((report.messages[2] as { role: string }).role).toBe("user");
  });

  it("deduplicates toolResult IDs seen across assistant spans", () => {
    // Two assistants with distinct tool call IDs. The second span contains
    // a duplicate toolResult for tc1 which was already consumed by the first.
    const messages: AgentMessage[] = [
      assistantWithToolCalls([{ id: "tc1", name: "bash", input: {} }]),
      toolResult("tc1", "bash"),
      assistantWithToolCalls([{ id: "tc2", name: "read_file", input: {} }]),
      toolResult("tc1", "bash"), // duplicate of tc1
      toolResult("tc2", "read_file"),
    ];
    const report = repairToolUseResultPairing(messages);
    // tc1 duplicate is detected and removed as orphan (not in tc2's toolCallIds)
    expect(report.messages).not.toBe(messages);
    // Both assistants should have their tool results
    const toolResults = report.messages.filter(
      (m) => (m as { role: string }).role === "toolResult",
    );
    expect(toolResults).toHaveLength(2);
  });

  it("drops orphan toolResult not associated with any assistant", () => {
    const messages: AgentMessage[] = [
      { role: "user", content: "hello" },
      toolResult("orphan1", "bash"),
    ];
    const report = repairToolUseResultPairing(messages);
    expect(report.droppedOrphanCount).toBe(1);
    expect(report.messages).toHaveLength(1);
    expect((report.messages[0] as { role: string }).role).toBe("user");
  });

  it("moves displaced toolResult back to correct position", () => {
    const messages: AgentMessage[] = [
      assistantWithToolCalls([{ id: "tc1", name: "bash", input: {} }]),
      { role: "user", content: "interruption" },
      toolResult("tc1", "bash"),
    ];
    const report = repairToolUseResultPairing(messages);
    expect(report.moved).toBe(true);
    // Tool result should be right after assistant, before user
    expect((report.messages[0] as { role: string }).role).toBe("assistant");
    expect((report.messages[1] as { role: string }).role).toBe("toolResult");
    expect((report.messages[2] as { role: string }).role).toBe("user");
  });

  it("skips synthetic results for errored/aborted assistant turns", () => {
    const messages: AgentMessage[] = [
      {
        ...assistantWithToolCalls([{ id: "tc1", name: "bash", input: {} }]),
        stopReason: "error",
      } as AgentMessage,
      { role: "user", content: "next" },
    ];
    const report = repairToolUseResultPairing(messages);
    // Should NOT insert synthetic result for errored turn
    expect(report.added).toHaveLength(0);
  });

  it("preserveErroredAssistantResults retains real results for errored turns", () => {
    const messages: AgentMessage[] = [
      {
        ...assistantWithToolCalls([{ id: "tc1", name: "bash", input: {} }]),
        stopReason: "error",
      } as AgentMessage,
      toolResult("tc1", "bash"),
    ];
    const report = repairToolUseResultPairing(messages, {
      preserveErroredAssistantResults: true,
    });
    const toolResults = report.messages.filter(
      (m) => (m as { role: string }).role === "toolResult",
    );
    expect(toolResults).toHaveLength(1);
  });

  it("handles multiple tool calls with partial results", () => {
    const messages: AgentMessage[] = [
      assistantWithToolCalls([
        { id: "tc1", name: "bash", input: {} },
        { id: "tc2", name: "read_file", input: {} },
      ]),
      toolResult("tc1", "bash"),
      // tc2 result missing
    ];
    const report = repairToolUseResultPairing(messages);
    expect(report.added).toHaveLength(1);
    expect(report.added[0].toolCallId).toBe("tc2");
    // Both results present after repair
    const toolResults = report.messages.filter(
      (m) => (m as { role: string }).role === "toolResult",
    );
    expect(toolResults).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// stripToolResultDetails
// ---------------------------------------------------------------------------

describe("stripToolResultDetails", () => {
  it("removes details property from toolResult messages", () => {
    const messages: AgentMessage[] = [
      {
        role: "toolResult",
        toolCallId: "tc1",
        content: [{ type: "text", text: "ok" }],
        details: { extra: "data" },
      } as unknown as AgentMessage,
    ];
    const result = stripToolResultDetails(messages);
    expect(result).not.toBe(messages);
    expect("details" in (result[0] as object)).toBe(false);
  });

  it("returns original array when no details present", () => {
    const messages: AgentMessage[] = [
      { role: "toolResult", toolCallId: "tc1", content: [{ type: "text", text: "ok" }] },
    ];
    const result = stripToolResultDetails(messages);
    expect(result).toBe(messages);
  });

  it("non-toolResult messages are unaffected", () => {
    const messages: AgentMessage[] = [
      { role: "user", content: "hello" },
      { role: "assistant", content: [{ type: "text", text: "hi" }] },
    ];
    const result = stripToolResultDetails(messages);
    expect(result).toBe(messages);
  });
});

// ---------------------------------------------------------------------------
// makeMissingToolResult
// ---------------------------------------------------------------------------

describe("makeMissingToolResult", () => {
  it("creates error toolResult with correct structure", () => {
    const result = makeMissingToolResult({ toolCallId: "tc1", toolName: "bash" });
    expect(result.role).toBe("toolResult");
    expect(result.toolCallId).toBe("tc1");
    expect(result.toolName).toBe("bash");
    expect(result.isError).toBe(true);
    expect(Array.isArray(result.content)).toBe(true);
  });

  it("falls back to 'unknown' tool name", () => {
    const result = makeMissingToolResult({ toolCallId: "tc1" });
    expect(result.toolName).toBe("unknown");
  });

  it("contains repair description in content", () => {
    const result = makeMissingToolResult({ toolCallId: "tc1" });
    const text = ((result.content as Array<{ text: string }>)[0]).text;
    expect(text).toContain("missing tool result");
    expect(text).toContain("transcript repair");
  });
});
