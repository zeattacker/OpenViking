import { describe, expect, it } from "vitest";

import {
  roughEstimate,
  msgTokenEstimate,
  convertToAgentMessages,
  normalizeAssistantContent,
  formatMessageFaithful,
  buildSystemPromptAddition,
} from "../context-engine.js";

// ---------------------------------------------------------------------------
// roughEstimate
// ---------------------------------------------------------------------------

describe("roughEstimate", () => {
  it("returns ceil(JSON.stringify(msgs).length / 4)", () => {
    const msgs = [{ role: "user", content: "hello" }];
    expect(roughEstimate(msgs)).toBe(Math.ceil(JSON.stringify(msgs).length / 4));
  });

  it("returns small number for empty array", () => {
    expect(roughEstimate([])).toBe(Math.ceil("[]".length / 4));
  });

  it("scales linearly with content", () => {
    const small = [{ role: "user", content: "hi" }];
    const large = [{ role: "user", content: "a".repeat(1000) }];
    expect(roughEstimate(large)).toBeGreaterThan(roughEstimate(small));
  });
});

// ---------------------------------------------------------------------------
// msgTokenEstimate
// ---------------------------------------------------------------------------

describe("msgTokenEstimate", () => {
  it("estimates string content as ceil(length / 4)", () => {
    expect(msgTokenEstimate({ role: "user", content: "hello world" })).toBe(
      Math.ceil("hello world".length / 4),
    );
  });

  it("estimates array content via JSON.stringify", () => {
    const content = [{ type: "text", text: "hello" }];
    expect(msgTokenEstimate({ role: "assistant", content })).toBe(
      Math.ceil(JSON.stringify(content).length / 4),
    );
  });

  it("returns 1 for missing content", () => {
    expect(msgTokenEstimate({ role: "user" })).toBe(1);
  });

  it("returns 1 for undefined content", () => {
    expect(msgTokenEstimate({ role: "user", content: undefined })).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// convertToAgentMessages
// ---------------------------------------------------------------------------

describe("convertToAgentMessages", () => {
  it("converts text parts to text content blocks", () => {
    const result = convertToAgentMessages({
      role: "assistant",
      parts: [{ type: "text", text: "hello" }],
    });
    expect(result).toHaveLength(1);
    expect(result[0].role).toBe("assistant");
    expect(result[0].content).toEqual([{ type: "text", text: "hello" }]);
  });

  it("converts tool parts with tool_id to toolUse + toolResult", () => {
    const result = convertToAgentMessages({
      role: "assistant",
      parts: [
        {
          type: "tool",
          tool_id: "tc1",
          tool_name: "bash",
          tool_input: { cmd: "ls" },
          tool_output: "file.txt",
          tool_status: "completed",
        },
      ],
    });
    expect(result).toHaveLength(2);
    // Assistant with toolUse
    expect(result[0].role).toBe("assistant");
    const content = result[0].content as Array<Record<string, unknown>>;
    expect(content[0]).toEqual({
      type: "toolUse",
      id: "tc1",
      name: "bash",
      input: { cmd: "ls" },
    });
    // toolResult
    const tr = result[1] as Record<string, unknown>;
    expect(tr.role).toBe("toolResult");
    expect(tr.toolCallId).toBe("tc1");
    expect(tr.isError).toBe(false);
  });

  it("marks error tool status as isError: true", () => {
    const result = convertToAgentMessages({
      role: "assistant",
      parts: [
        {
          type: "tool",
          tool_id: "tc1",
          tool_name: "bash",
          tool_input: {},
          tool_output: "error msg",
          tool_status: "error",
        },
      ],
    });
    const tr = result[1] as Record<string, unknown>;
    expect(tr.isError).toBe(true);
  });

  it("creates interrupted result for running tool", () => {
    const result = convertToAgentMessages({
      role: "assistant",
      parts: [
        {
          type: "tool",
          tool_id: "tc1",
          tool_name: "bash",
          tool_input: {},
          tool_output: "",
          tool_status: "running",
        },
      ],
    });
    const tr = result[1] as Record<string, unknown>;
    expect(tr.isError).toBe(false);
    const text = ((tr.content as Array<{ text: string }>)[0]).text;
    expect(text).toContain("interrupted");
  });

  it("degrades tool parts without tool_id to text blocks", () => {
    const result = convertToAgentMessages({
      role: "assistant",
      parts: [
        {
          type: "tool",
          tool_id: "",
          tool_name: "grep",
          tool_input: { pattern: "TODO" },
          tool_output: "found 3",
          tool_status: "completed",
        },
      ],
    });
    expect(result).toHaveLength(1);
    const content = result[0].content as Array<{ type: string; text: string }>;
    expect(content[0].type).toBe("text");
    expect(content[0].text).toContain("[grep]");
    expect(content[0].text).toContain("Output: found 3");
  });

  it("converts context parts with abstract to text", () => {
    const result = convertToAgentMessages({
      role: "assistant",
      parts: [{ type: "context", abstract: "User likes TypeScript" }],
    });
    expect(result).toHaveLength(1);
    const content = result[0].content as Array<{ type: string; text: string }>;
    expect(content[0]).toEqual({ type: "text", text: "User likes TypeScript" });
  });

  it("joins text parts for non-assistant messages", () => {
    const result = convertToAgentMessages({
      role: "user",
      parts: [
        { type: "text", text: "line one" },
        { type: "text", text: "line two" },
      ],
    });
    expect(result).toHaveLength(1);
    expect(result[0].role).toBe("user");
    expect(result[0].content).toBe("line one\nline two");
  });

  it("returns empty string content for user with no parts", () => {
    const result = convertToAgentMessages({ role: "user", parts: [] });
    expect(result).toHaveLength(1);
    expect(result[0].content).toBe("");
  });
});

// ---------------------------------------------------------------------------
// normalizeAssistantContent
// ---------------------------------------------------------------------------

describe("normalizeAssistantContent", () => {
  it("wraps string assistant content into array", () => {
    const messages = [{ role: "assistant", content: "hello" }];
    normalizeAssistantContent(messages);
    expect(messages[0].content).toEqual([{ type: "text", text: "hello" }]);
  });

  it("leaves array content unchanged", () => {
    const content = [{ type: "text", text: "hello" }];
    const messages = [{ role: "assistant", content }];
    normalizeAssistantContent(messages);
    expect(messages[0].content).toBe(content);
  });

  it("does not modify user messages", () => {
    const messages = [{ role: "user", content: "hello" }];
    normalizeAssistantContent(messages);
    expect(messages[0].content).toBe("hello");
  });
});

// ---------------------------------------------------------------------------
// formatMessageFaithful
// ---------------------------------------------------------------------------

describe("formatMessageFaithful", () => {
  it("formats text parts", () => {
    const result = formatMessageFaithful({
      id: "1",
      role: "assistant",
      created_at: "",
      parts: [{ type: "text", text: "Hello world" }],
    });
    expect(result).toContain("[assistant]:");
    expect(result).toContain("Hello world");
  });

  it("formats tool parts with status and output", () => {
    const result = formatMessageFaithful({
      id: "1",
      role: "assistant",
      created_at: "",
      parts: [
        {
          type: "tool",
          tool_name: "bash",
          tool_status: "completed",
          tool_input: { cmd: "ls" },
          tool_output: "file.txt",
        },
      ],
    });
    expect(result).toContain("[Tool: bash]");
    expect(result).toContain("(completed)");
    expect(result).toContain("file.txt");
  });

  it("formats context parts with URI", () => {
    const result = formatMessageFaithful({
      id: "1",
      role: "assistant",
      created_at: "",
      parts: [{ type: "context", uri: "viking://test", abstract: "summary" }],
    });
    expect(result).toContain("[Context: viking://test]");
    expect(result).toContain("summary");
  });

  it("handles empty parts", () => {
    const result = formatMessageFaithful({
      id: "1",
      role: "user",
      created_at: "",
      parts: [],
    });
    expect(result).toContain("[user]:");
    expect(result).toContain("(empty)");
  });
});

// ---------------------------------------------------------------------------
// buildSystemPromptAddition
// ---------------------------------------------------------------------------

describe("buildSystemPromptAddition", () => {
  it("contains Compressed Context section header", () => {
    const result = buildSystemPromptAddition();
    expect(result).toContain("Session Context Guide");
  });

  it("references ov_archive_expand tool", () => {
    const result = buildSystemPromptAddition();
    expect(result).toContain("ov_archive_expand");
  });

  it("mentions Archive Index", () => {
    const result = buildSystemPromptAddition();
    expect(result).toContain("Archive Index");
  });

  it("mentions Session History Summary", () => {
    const result = buildSystemPromptAddition();
    expect(result).toContain("Session History Summary");
  });
});
