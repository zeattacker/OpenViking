/**
 * P2: Drift Detector for OpenViking plugin.
 *
 * Tracks alignment scores in a sliding window and raises alerts
 * when sustained degradation is detected.
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";

import type { AlignmentResult } from "./alignment.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DriftState {
  window: number[];
  consecutiveFlags: number;
  totalEvaluated: number;
  totalFlagged: number;
  lastAlert: string | null;
}

export interface DriftAlert {
  type: "drift_alert";
  mean: number;
  consecutiveFlags: number;
  windowSize: number;
  totalEvaluated: number;
}

type Logger = {
  info: (msg: string) => void;
  warn?: (msg: string) => void;
};

// ---------------------------------------------------------------------------
// DriftDetector
// ---------------------------------------------------------------------------

export class DriftDetector {
  private state: DriftState;
  private readonly windowSize: number;
  private readonly alertThreshold: number;
  private readonly consecutiveFlagLimit: number;
  private readonly statePath: string;
  private readonly logger: Logger;
  private loaded = false;

  constructor(config: {
    windowSize: number;
    alertThreshold: number;
    consecutiveFlagLimit: number;
    dataDir: string;
    logger: Logger;
  }) {
    this.windowSize = config.windowSize;
    this.alertThreshold = config.alertThreshold;
    this.consecutiveFlagLimit = config.consecutiveFlagLimit;
    this.statePath = join(config.dataDir, "alignment", "drift_state.json");
    this.logger = config.logger;
    this.state = {
      window: [],
      consecutiveFlags: 0,
      totalEvaluated: 0,
      totalFlagged: 0,
      lastAlert: null,
    };
  }

  record(result: AlignmentResult): DriftAlert | null {
    if (!this.loaded) {
      this.loadState();
      this.loaded = true;
    }

    this.state.totalEvaluated++;
    this.state.window.push(result.score);

    while (this.state.window.length > this.windowSize) {
      this.state.window.shift();
    }

    if (result.verdict !== "pass") {
      this.state.consecutiveFlags++;
      this.state.totalFlagged++;
    } else {
      this.state.consecutiveFlags = 0;
    }

    // Check alert conditions
    let alert: DriftAlert | null = null;

    if (this.state.window.length >= 5) {
      const mean =
        this.state.window.reduce((a, b) => a + b, 0) /
        this.state.window.length;

      if (
        mean < this.alertThreshold ||
        this.state.consecutiveFlags >= this.consecutiveFlagLimit
      ) {
        alert = {
          type: "drift_alert",
          mean,
          consecutiveFlags: this.state.consecutiveFlags,
          windowSize: this.state.window.length,
          totalEvaluated: this.state.totalEvaluated,
        };
        this.state.lastAlert = new Date().toISOString();
      }
    }

    this.saveState();
    return alert;
  }

  getState(): DriftState {
    if (!this.loaded) {
      this.loadState();
      this.loaded = true;
    }
    return { ...this.state, window: [...this.state.window] };
  }

  reset(): void {
    this.state = {
      window: [],
      consecutiveFlags: 0,
      totalEvaluated: 0,
      totalFlagged: 0,
      lastAlert: null,
    };
    this.saveState();
  }

  private loadState(): void {
    try {
      const raw = readFileSync(this.statePath, "utf-8");
      const parsed = JSON.parse(raw) as Partial<DriftState>;
      if (Array.isArray(parsed.window)) {
        this.state.window = parsed.window.filter(
          (v) => typeof v === "number" && Number.isFinite(v),
        );
      }
      if (typeof parsed.consecutiveFlags === "number")
        this.state.consecutiveFlags = parsed.consecutiveFlags;
      if (typeof parsed.totalEvaluated === "number")
        this.state.totalEvaluated = parsed.totalEvaluated;
      if (typeof parsed.totalFlagged === "number")
        this.state.totalFlagged = parsed.totalFlagged;
      if (typeof parsed.lastAlert === "string")
        this.state.lastAlert = parsed.lastAlert;
    } catch {
      // No state file or parse error — start fresh
    }
  }

  private saveState(): void {
    try {
      mkdirSync(dirname(this.statePath), { recursive: true });
      writeFileSync(this.statePath, JSON.stringify(this.state), "utf-8");
    } catch {
      // Non-fatal — state will be rebuilt on next load
    }
  }
}
