import type { Recommendation } from "../api";

const SEVERITY_COLOR: Record<Recommendation["severity"], string> = {
  CRITICAL: "border-l-rose-600 bg-rose-50",
  HIGH: "border-l-rose-500 bg-rose-50",
  MEDIUM: "border-l-amber-500 bg-amber-50",
  LOW: "border-l-emerald-500 bg-emerald-50",
};

export function RecommendationCard({ rec }: { rec: Recommendation }) {
  return (
    <div
      className={`rounded-md border border-l-4 ${SEVERITY_COLOR[rec.severity]} border-slate-200 p-4`}
    >
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-900">{rec.title}</h3>
        <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
          {rec.severity} · {rec.category}
        </span>
      </div>
      <p className="mb-3 text-sm text-slate-700">{rec.description}</p>
      <ul className="ml-4 list-disc space-y-1 text-sm text-slate-700">
        {rec.actions.map((a, i) => (
          <li key={i}>{a}</li>
        ))}
      </ul>
      {rec.estimated_impact && (
        <p className="mt-3 text-xs italic text-slate-500">{rec.estimated_impact}</p>
      )}
    </div>
  );
}
