import { describe, expect, it } from "vitest";

import {
  sanitizeToolCallId,
  extractToolCallsFromAssistant,
  extractToolResultId,
  isValidCloudCodeAssistToolId,
  sanitizeToolCallIdsForCloudCodeAssist,
} from "../tool-call-id.js";
import type { AgentMessage } from "../tool-call-id.js";

// ---------------------------------------------------------------------------
// sanitizeToolCallId
// ---------------------------------------------------------------------------

describe("sanitizeToolCallId", () => {
  describe("strict mode", () => {
    it("passes through alphanumeric ID unchanged", () => {
      expect(sanitizeToolCallId("abc123")).toBe("abc123");
    });

    it("strips non-alphanumeric characters", () => {
      expect(sanitizeToolCallId("toolu_abc-123")).toBe("tooluabc123");
    });

    it("returns 'defaulttoolid' for empty string", () => {
      expect(sanitizeToolCallId("")).toBe("defaulttoolid");
    });

    it("returns 'sanitizedtoolid' when all chars are stripped", () => {
      expect(sanitizeToolCallId("---")).toBe("sanitizedtoolid");
    });

    it("returns 'defaulttoolid' for non-string input", () => {
      expect(sanitizeToolCallId(undefined as unknown as string)).toBe("defaulttoolid");
      expect(sanitizeToolCallId(null as unknown as string)).toBe("defaulttoolid");
    });
  });

  describe("strict9 mode", () => {
    it("truncates long alphanumeric ID to 9 chars", () => {
      expect(sanitizeToolCallId("abcdefghijklm", "strict9")).toBe("abcdefghi");
    });

    it("hashes short alphanumeric ID to 9 chars", () => {
      const result = sanitizeToolCallId("ab", "strict9");
      expect(result).toHaveLength(9);
      expect(/^[a-f0-9]+$/.test(result)).toBe(true);
    });

    it("returns 'defaultid' for empty string", () => {
      expect(sanitizeToolCallId("", "strict9")).toBe("defaultid");
    });

    it("produces 9-char hash when all chars are stripped", () => {
      const result = sanitizeToolCallId("---", "strict9");
      expect(result).toHaveLength(9);
    });

    it("exactly 9 alphanumeric chars passes through", () => {
      expect(sanitizeToolCallId("abcdefghi", "strict9")).toBe("abcdefghi");
    });
  });
});

// ---------------------------------------------------------------------------
// isValidCloudCodeAssistToolId
// ---------------------------------------------------------------------------

describe("isValidCloudCodeAssistToolId", () => {
  it("accepts valid alphanumeric in strict mode", () => {
    expect(isValidCloudCodeAssistToolId("abc123")).toBe(true);
  });

  it("rejects special chars in strict mode", () => {
    expect(isValidCloudCodeAssistToolId("abc-123")).toBe(false);
    expect(isValidCloudCodeAssistToolId("abc_123")).toBe(false);
  });

  it("rejects empty string", () => {
    expect(isValidCloudCodeAssistToolId("")).toBe(false);
  });

  it("accepts 9-char alphanumeric in strict9 mode", () => {
    expect(isValidCloudCodeAssistToolId("abcdefghi", "strict9")).toBe(true);
  });

  it("rejects 8-char in strict9 mode", () => {
    expect(isValidCloudCodeAssistToolId("abcdefgh", "strict9")).toBe(false);
  });

  it("rejects 10-char in strict9 mode", () => {
    expect(isValidCloudCodeAssistToolId("abcdefghij", "strict9")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// extractToolCallsFromAssistant
// ---------------------------------------------------------------------------

describe("extractToolCallsFromAssistant", () => {
  it("extracts from toolCall blocks", () => {
    const msg = {
      role: "assistant" as const,
      content: [
        { type: "toolCall", id: "tc1", name: "read_file" },
      ],
    };
    expect(extractToolCallsFromAssistant(msg)).toEqual([
      { id: "tc1", name: "read_file" },
    ]);
  });

  it("extracts from toolUse blocks", () => {
    const msg = {
      role: "assistant" as const,
      content: [{ type: "toolUse", id: "tu1", name: "bash" }],
    };
    expect(extractToolCallsFromAssistant(msg)).toEqual([
      { id: "tu1", name: "bash" },
    ]);
  });

  it("extracts from functionCall blocks", () => {
    const msg = {
      role: "assistant" as const,
      content: [{ type: "functionCall", id: "fc1", name: "search" }],
    };
    expect(extractToolCallsFromAssistant(msg)).toEqual([
      { id: "fc1", name: "search" },
    ]);
  });

  it("skips blocks with no id", () => {
    const msg = {
      role: "assistant" as const,
      content: [
        { type: "toolCall", name: "read_file" },
        { type: "toolCall", id: "", name: "bash" },
      ],
    };
    expect(extractToolCallsFromAssistant(msg)).toEqual([]);
  });

  it("returns empty for non-array content", () => {
    const msg = { role: "assistant" as const, content: "just text" };
    expect(extractToolCallsFromAssistant(msg)).toEqual([]);
  });

  it("returns empty for text-only content blocks", () => {
    const msg = {
      role: "assistant" as const,
      content: [{ type: "text", text: "hello" }],
    };
    expect(extractToolCallsFromAssistant(msg)).toEqual([]);
  });

  it("extracts multiple tool calls", () => {
    const msg = {
      role: "assistant" as const,
      content: [
        { type: "text", text: "Let me check" },
        { type: "toolCall", id: "tc1", name: "read_file" },
        { type: "toolUse", id: "tc2", name: "bash" },
      ],
    };
    expect(extractToolCallsFromAssistant(msg)).toEqual([
      { id: "tc1", name: "read_file" },
      { id: "tc2", name: "bash" },
    ]);
  });
});

// ---------------------------------------------------------------------------
// extractToolResultId
// ---------------------------------------------------------------------------

describe("extractToolResultId", () => {
  it("returns toolCallId when present", () => {
    const msg = { role: "toolResult" as const, toolCallId: "tc1" };
    expect(extractToolResultId(msg)).toBe("tc1");
  });

  it("falls back to toolUseId when no toolCallId", () => {
    const msg = { role: "toolResult" as const, toolUseId: "tu1" };
    expect(extractToolResultId(msg)).toBe("tu1");
  });

  it("returns null when neither is present", () => {
    const msg = { role: "toolResult" as const };
    expect(extractToolResultId(msg)).toBeNull();
  });

  it("prefers toolCallId over toolUseId", () => {
    const msg = { role: "toolResult" as const, toolCallId: "tc1", toolUseId: "tu1" };
    expect(extractToolResultId(msg)).toBe("tc1");
  });

  it("returns null for empty string values", () => {
    const msg = { role: "toolResult" as const, toolCallId: "", toolUseId: "" };
    expect(extractToolResultId(msg)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// sanitizeToolCallIdsForCloudCodeAssist
// ---------------------------------------------------------------------------

describe("sanitizeToolCallIdsForCloudCodeAssist", () => {
  it("returns original array reference when no changes needed", () => {
    const messages: AgentMessage[] = [
      { role: "user", content: "hello" },
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "abc123", name: "read" }],
      },
      { role: "toolResult", toolCallId: "abc123", content: [{ type: "text", text: "ok" }] },
    ];
    const result = sanitizeToolCallIdsForCloudCodeAssist(messages);
    expect(result).toBe(messages);
  });

  it("rewrites non-alphanumeric IDs in assistant and toolResult", () => {
    const messages: AgentMessage[] = [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "toolu_abc-123", name: "bash" }],
      },
      { role: "toolResult", toolCallId: "toolu_abc-123" },
    ];
    const result = sanitizeToolCallIdsForCloudCodeAssist(messages);
    expect(result).not.toBe(messages);
    const assistantContent = (result[0] as { content: Array<{ id: string }> }).content;
    expect(assistantContent[0].id).toBe("tooluabc123");
    expect((result[1] as { toolCallId: string }).toolCallId).toBe("tooluabc123");
  });

  it("user messages pass through unchanged", () => {
    const userMsg: AgentMessage = { role: "user", content: "hello" };
    const messages: AgentMessage[] = [userMsg];
    const result = sanitizeToolCallIdsForCloudCodeAssist(messages);
    expect(result).toBe(messages);
  });

  it("strict9 mode produces 9-char IDs", () => {
    const messages: AgentMessage[] = [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "very-long-id-here", name: "bash" }],
      },
      { role: "toolResult", toolCallId: "very-long-id-here" },
    ];
    const result = sanitizeToolCallIdsForCloudCodeAssist(messages, "strict9");
    const assistantContent = (result[0] as { content: Array<{ id: string }> }).content;
    expect(assistantContent[0].id).toHaveLength(9);
    expect(/^[a-zA-Z0-9]{9}$/.test(assistantContent[0].id)).toBe(true);
  });

  it("handles duplicate raw IDs with occurrence-aware resolution", () => {
    const messages: AgentMessage[] = [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "toolu_abc", name: "bash" }],
      },
      { role: "toolResult", toolCallId: "toolu_abc" },
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "toolu_abc", name: "read" }],
      },
      { role: "toolResult", toolCallId: "toolu_abc" },
    ];
    const result = sanitizeToolCallIdsForCloudCodeAssist(messages);
    const id1 = ((result[0] as { content: Array<{ id: string }> }).content)[0].id;
    const id2 = ((result[2] as { content: Array<{ id: string }> }).content)[0].id;
    // Both sanitized, but distinct
    expect(id1).not.toBe(id2);
    // Tool results match their respective assistant IDs
    expect((result[1] as { toolCallId: string }).toolCallId).toBe(id1);
    expect((result[3] as { toolCallId: string }).toolCallId).toBe(id2);
  });
});
