/**
 * P2: Alignment Check for OpenViking plugin.
 *
 * Evaluates agent responses against extracted constraints (regex patterns + keywords).
 * Three layers: pattern match (L1), keyword proximity (L2), LLM evaluation (L3, stub for v1).
 */

import { createHash } from "node:crypto";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface StructuredConstraint {
  id: string;
  type: "hard_block" | "soft_flag";
  description: string;
  patterns: string[];
  keywords: string[];
  source: "system_default" | "agent_instructions";
}

export interface AlignmentProfile {
  constraints: StructuredConstraint[];
  instructionsHash: string;
}

export interface AlignmentIssue {
  constraintId: string;
  type: "hard_block" | "soft_flag";
  description: string;
  matchedText?: string;
  layer: 1 | 2 | 3;
}

export interface AlignmentResult {
  verdict: "pass" | "soft_flag" | "hard_block";
  issues: AlignmentIssue[];
  score: number; // 1.0 = pass, 0.5 = soft_flag, 0.0 = hard_block
  timestamp: number;
}

// ---------------------------------------------------------------------------
// Default safety constraints (always active)
// ---------------------------------------------------------------------------

const DEFAULT_CONSTRAINTS: StructuredConstraint[] = [
  {
    id: "no_credential_leak",
    type: "hard_block",
    description: "Never expose API keys, passwords, or secrets in responses",
    patterns: [
      "(api[_\\-]?key|password|secret|token|bearer)\\s*[:=]\\s*['\"][^'\"]{8,}",
      "sk-[a-zA-Z0-9]{20,}",
      "ghp_[a-zA-Z0-9]{36,}",
      "AKIA[A-Z0-9]{16}",
    ],
    keywords: ["api_key", "password", "secret", "bearer", "credential"],
    source: "system_default",
  },
  {
    id: "no_pii_exposure",
    type: "soft_flag",
    description: "Avoid exposing personal identifiable information",
    patterns: [
      "\\b\\d{3}-\\d{2}-\\d{4}\\b", // SSN
      "\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b", // Credit card
    ],
    keywords: ["social security", "credit card", "passport number"],
    source: "system_default",
  },
  {
    id: "no_harmful_instructions",
    type: "hard_block",
    description: "Never provide instructions for creating weapons, malware, or illegal activities",
    patterns: [
      "(?:step[\\s-]*by[\\s-]*step|how\\s+to|instructions?\\s+(?:for|to))\\s+(?:make|create|build|construct)\\s+(?:a\\s+)?(?:bomb|weapon|explosive|malware|virus|ransomware)",
    ],
    keywords: ["synthesize explosive", "build weapon", "create malware", "bypass security"],
    source: "system_default",
  },
];

// ---------------------------------------------------------------------------
// Profile cache
// ---------------------------------------------------------------------------

const profileCache = new Map<
  string,
  { profile: AlignmentProfile; expires: number }
>();

const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes

// ---------------------------------------------------------------------------
// Constraint extraction (heuristic, v1)
// ---------------------------------------------------------------------------

/**
 * Extract structured constraints from free-text agent instructions using
 * heuristic pattern matching. Looks for imperative rules like "never",
 * "must not", "do not", "always", etc.
 */
function extractConstraintsFromInstructions(
  text: string,
): StructuredConstraint[] {
  if (!text || text.trim().length < 10) return [];

  const constraints: StructuredConstraint[] = [];
  const lines = text.split(/[.\n]+/);
  let idx = 0;

  const hardPatterns = [
    /\b(?:never|must\s+not|shall\s+not|do\s+not|don['']t|cannot|prohibited|forbidden)\b/i,
  ];
  const softPatterns = [
    /\b(?:avoid|prefer\s+not|should\s+not|shouldn['']t|try\s+not|discouraged|refrain)\b/i,
  ];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (line.length < 10 || line.length > 500) continue;

    const isHard = hardPatterns.some((p) => p.test(line));
    const isSoft = !isHard && softPatterns.some((p) => p.test(line));

    if (!isHard && !isSoft) continue;

    idx++;
    const id = `agent_constraint_${idx}`;

    // Extract key nouns/phrases as keywords (simple word extraction)
    const keywords = line
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 3)
      .slice(0, 5);

    constraints.push({
      id,
      type: isHard ? "hard_block" : "soft_flag",
      description: line.slice(0, 200),
      patterns: [], // No auto-generated regex for v1 — rely on keyword matching
      keywords,
      source: "agent_instructions",
    });
  }

  return constraints;
}

// ---------------------------------------------------------------------------
// Profile assembly
// ---------------------------------------------------------------------------

export function assembleProfile(instructionsText: string): AlignmentProfile {
  const hash = createHash("md5")
    .update(instructionsText || "")
    .digest("hex")
    .slice(0, 16);

  // Check cache
  const cached = profileCache.get(hash);
  if (cached && cached.expires > Date.now()) {
    return cached.profile;
  }

  const agentConstraints = extractConstraintsFromInstructions(instructionsText);
  const profile: AlignmentProfile = {
    constraints: [...DEFAULT_CONSTRAINTS, ...agentConstraints],
    instructionsHash: hash,
  };

  profileCache.set(hash, { profile, expires: Date.now() + CACHE_TTL_MS });
  return profile;
}

// ---------------------------------------------------------------------------
// Layer 1: Pattern-based check (zero latency)
// ---------------------------------------------------------------------------

function checkPatterns(
  response: string,
  constraints: StructuredConstraint[],
): AlignmentIssue[] {
  const issues: AlignmentIssue[] = [];

  for (const constraint of constraints) {
    for (const patternStr of constraint.patterns) {
      try {
        const regex = new RegExp(patternStr, "i");
        const match = regex.exec(response);
        if (match) {
          issues.push({
            constraintId: constraint.id,
            type: constraint.type,
            description: constraint.description,
            matchedText: match[0].slice(0, 100),
            layer: 1,
          });
          break; // One match per constraint is enough
        }
      } catch {
        // Invalid regex — skip
      }
    }
  }

  return issues;
}

// ---------------------------------------------------------------------------
// Layer 2: Keyword proximity check (zero latency)
// ---------------------------------------------------------------------------

function checkKeywords(
  response: string,
  constraints: StructuredConstraint[],
): AlignmentIssue[] {
  const issues: AlignmentIssue[] = [];
  const lowerResponse = response.toLowerCase();

  for (const constraint of constraints) {
    if (constraint.keywords.length === 0) continue;

    // Require at least 2 keyword matches for agent constraints (reduce false positives)
    const threshold = constraint.source === "agent_instructions" ? 2 : 1;
    let matchCount = 0;
    const matched: string[] = [];

    for (const keyword of constraint.keywords) {
      if (lowerResponse.includes(keyword.toLowerCase())) {
        matchCount++;
        matched.push(keyword);
      }
    }

    if (matchCount >= threshold) {
      issues.push({
        constraintId: constraint.id,
        type: constraint.type,
        description: constraint.description,
        matchedText: `keywords: ${matched.join(", ")}`,
        layer: 2,
      });
    }
  }

  return issues;
}

// ---------------------------------------------------------------------------
// Main alignment check
// ---------------------------------------------------------------------------

export function alignmentCheck(
  response: string,
  profile: AlignmentProfile,
): AlignmentResult {
  const timestamp = Date.now();

  // Layer 1: Pattern match
  const patternIssues = checkPatterns(response, profile.constraints);
  const hasHardBlock = patternIssues.some((i) => i.type === "hard_block");
  if (hasHardBlock) {
    return {
      verdict: "hard_block",
      issues: patternIssues,
      score: 0.0,
      timestamp,
    };
  }

  // Layer 2: Keyword proximity
  const keywordIssues = checkKeywords(response, profile.constraints);
  const allIssues = [...patternIssues, ...keywordIssues];

  // Deduplicate by constraintId (pattern match takes priority)
  const seen = new Set<string>();
  const dedupedIssues: AlignmentIssue[] = [];
  for (const issue of allIssues) {
    if (!seen.has(issue.constraintId)) {
      seen.add(issue.constraintId);
      dedupedIssues.push(issue);
    }
  }

  if (dedupedIssues.some((i) => i.type === "hard_block")) {
    return {
      verdict: "hard_block",
      issues: dedupedIssues,
      score: 0.0,
      timestamp,
    };
  }

  if (dedupedIssues.length > 0) {
    return {
      verdict: "soft_flag",
      issues: dedupedIssues,
      score: 0.5,
      timestamp,
    };
  }

  return { verdict: "pass", issues: [], score: 1.0, timestamp };
}

// ---------------------------------------------------------------------------
// Layer 3: LLM evaluation (stub for v1)
// ---------------------------------------------------------------------------

export async function llmAlignmentCheck(
  _response: string,
  _profile: AlignmentProfile,
): Promise<AlignmentResult> {
  // v1 stub — always passes. Implement with OpenViking VLM in v2.
  return { verdict: "pass", issues: [], score: 1.0, timestamp: Date.now() };
}
