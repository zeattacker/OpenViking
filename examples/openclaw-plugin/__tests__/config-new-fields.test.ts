import { describe, expect, it } from "vitest";

import { memoryOpenVikingConfigSchema } from "../config.js";

const parse = memoryOpenVikingConfigSchema.parse;

describe("config new fields", () => {
  // -------------------------------------------------------------------
  // emitStandardDiagnostics
  // -------------------------------------------------------------------

  describe("emitStandardDiagnostics", () => {
    it("defaults to false", () => {
      expect(parse({}).emitStandardDiagnostics).toBe(false);
    });

    it("accepts true", () => {
      expect(parse({ emitStandardDiagnostics: true }).emitStandardDiagnostics).toBe(true);
    });

    it("accepts false explicitly", () => {
      expect(parse({ emitStandardDiagnostics: false }).emitStandardDiagnostics).toBe(false);
    });
  });

  // -------------------------------------------------------------------
  // commitTokenThreshold
  // -------------------------------------------------------------------

  describe("commitTokenThreshold", () => {
    it("defaults to 20000", () => {
      expect(parse({}).commitTokenThreshold).toBe(20000);
    });

    it("accepts 0 (commit every turn)", () => {
      expect(parse({ commitTokenThreshold: 0 }).commitTokenThreshold).toBe(0);
    });

    it("clamps to min 0", () => {
      expect(parse({ commitTokenThreshold: -100 }).commitTokenThreshold).toBe(0);
    });

    it("clamps to max 100000", () => {
      expect(parse({ commitTokenThreshold: 999999 }).commitTokenThreshold).toBe(100000);
    });

    it("floors fractional values", () => {
      expect(parse({ commitTokenThreshold: 1500.7 }).commitTokenThreshold).toBe(1500);
    });
  });

  // -------------------------------------------------------------------
  // alignment
  // -------------------------------------------------------------------

  describe("alignment", () => {
    it("defaults to disabled with observe_only mode", () => {
      const result = parse({});
      expect(result.alignment.enabled).toBe(false);
      expect(result.alignment.mode).toBe("observe_only");
    });

    it("accepts enabled: true", () => {
      expect(parse({ alignment: { enabled: true } }).alignment.enabled).toBe(true);
    });

    it("accepts valid modes", () => {
      expect(parse({ alignment: { mode: "soft_enforce" } }).alignment.mode).toBe("soft_enforce");
      expect(parse({ alignment: { mode: "full_enforce" } }).alignment.mode).toBe("full_enforce");
    });

    it("falls back to observe_only for invalid mode", () => {
      expect(parse({ alignment: { mode: "invalid" } }).alignment.mode).toBe("observe_only");
    });

    it("defaults llmCheckThreshold to 500", () => {
      expect(parse({}).alignment.llmCheckThreshold).toBe(500);
    });

    it("clamps driftAlertThreshold between 0 and 1", () => {
      expect(parse({ alignment: { driftAlertThreshold: -1 } }).alignment.driftAlertThreshold).toBe(0);
      expect(parse({ alignment: { driftAlertThreshold: 2 } }).alignment.driftAlertThreshold).toBe(1);
    });
  });

  // -------------------------------------------------------------------
  // profileInjection
  // -------------------------------------------------------------------

  describe("profileInjection", () => {
    it("defaults to true", () => {
      expect(parse({}).profileInjection).toBe(true);
    });

    it("accepts false", () => {
      expect(parse({ profileInjection: false }).profileInjection).toBe(false);
    });
  });

  // -------------------------------------------------------------------
  // recallFormat
  // -------------------------------------------------------------------

  describe("recallFormat", () => {
    it("defaults to function_call", () => {
      expect(parse({}).recallFormat).toBe("function_call");
    });

    it("accepts xml", () => {
      expect(parse({ recallFormat: "xml" }).recallFormat).toBe("xml");
    });

    it("falls back to function_call for invalid value", () => {
      expect(parse({ recallFormat: "invalid" }).recallFormat).toBe("function_call");
    });
  });

  // -------------------------------------------------------------------
  // Unknown keys
  // -------------------------------------------------------------------

  describe("unknown keys", () => {
    it("throws on unknown config keys", () => {
      expect(() => parse({ unknownKey: true })).toThrow("unknown keys");
    });
  });
});
