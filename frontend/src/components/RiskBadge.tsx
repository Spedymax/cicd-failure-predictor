import type { Decision } from "../api";

const STYLES: Record<Decision, string> = {
  auto_approve: "bg-emerald-100 text-emerald-800 border-emerald-200",
  warn: "bg-amber-100 text-amber-800 border-amber-200",
  block: "bg-rose-100 text-rose-800 border-rose-200",
};

const LABELS: Record<Decision, string> = {
  auto_approve: "AUTO_APPROVE",
  warn: "WARN",
  block: "BLOCK",
};

export function RiskBadge({ decision, score }: { decision: Decision; score: number }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${STYLES[decision]}`}
    >
      {LABELS[decision]} · {(score * 100).toFixed(0)}%
    </span>
  );
}
