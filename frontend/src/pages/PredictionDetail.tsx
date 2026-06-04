import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { Decision, fetchPrediction, overridePrediction } from "../api";
import { ClassChip } from "../components/ClassChip";
import {
  FeatureImportanceChart,
  FeatureImportanceChartLegacy,
} from "../components/FeatureImportanceChart";
import { RecommendationCard } from "../components/RecommendationCard";
import { RiskBadge } from "../components/RiskBadge";

const DECISION_OPTIONS: { value: Decision; label: string; tone: string }[] = [
  { value: "auto_approve", label: "Auto approve", tone: "bg-emerald-100 text-emerald-800" },
  { value: "warn", label: "Warn", tone: "bg-amber-100 text-amber-800" },
  { value: "block", label: "Block", tone: "bg-rose-100 text-rose-800" },
];

function OverrideModal({
  predictionId,
  currentDecision,
  onClose,
}: {
  predictionId: number;
  currentDecision: Decision;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [newDecision, setNewDecision] = useState<Decision>(currentDecision);
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: () => overridePrediction(predictionId, newDecision, reason),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["prediction", predictionId] });
      onClose();
    },
    onError: (err: unknown) => {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(msg ?? "Override failed");
    },
  });
  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-slate-900/60 p-4">
      <div className="w-full max-w-md space-y-4 rounded-lg bg-white p-6 shadow-xl">
        <h2 className="text-lg font-bold text-slate-900">Override decision</h2>
        <div>
          <p className="mb-2 text-xs uppercase tracking-wide text-slate-500">New decision</p>
          <div className="flex gap-2">
            {DECISION_OPTIONS.map((o) => (
              <button
                key={o.value}
                onClick={() => setNewDecision(o.value)}
                className={`flex-1 rounded border px-3 py-2 text-sm font-medium ${
                  newDecision === o.value
                    ? `${o.tone} border-current`
                    : "border-slate-200 text-slate-600 hover:bg-slate-50"
                }`}
              >
                {o.label}
              </button>
            ))}
          </div>
        </div>
        <label className="block text-sm">
          <span className="text-xs uppercase tracking-wide text-slate-500">Reason</span>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            required
            maxLength={1000}
            rows={3}
            placeholder="Чому переписуємо рішення моделі?"
            className="mt-1 w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />
        </label>
        {error && <p className="text-sm text-rose-600">{error}</p>}
        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded border border-slate-300 px-4 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
          >
            Cancel
          </button>
          <button
            disabled={!reason || newDecision === currentDecision || mutation.isPending}
            onClick={() => mutation.mutate()}
            className="rounded bg-slate-900 px-4 py-1.5 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
          >
            {mutation.isPending ? "Submitting..." : "Override"}
          </button>
        </div>
      </div>
    </div>
  );
}

function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-slate-200 bg-white p-3">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-base font-semibold text-slate-900">{value}</p>
    </div>
  );
}

export default function PredictionDetail() {
  const { id } = useParams<{ id: string }>();
  const predictionId = Number(id);
  const { data, isLoading, error } = useQuery({
    queryKey: ["prediction", predictionId],
    queryFn: () => fetchPrediction(predictionId),
    enabled: !Number.isNaN(predictionId),
  });
  const [showOverride, setShowOverride] = useState(false);

  if (isLoading) return <p className="p-6 text-slate-500">Завантаження...</p>;
  if (error || !data) return <p className="p-6 text-rose-600">Не знайдено</p>;

  const sortedProbs = Object.entries(data.class_probabilities).sort(([, a], [, b]) => b - a);

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-8 py-8">
      <Link to="/" className="text-sm text-blue-600 hover:underline">
        ← Усі прогнози
      </Link>
      <header className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">
              Prediction #{data.id}
              {data.repository_full_name && (
                <>
                  {" · "}
                  <span className="font-mono normal-case text-slate-700">
                    {data.repository_full_name}
                  </span>
                </>
              )}
            </p>
            <h1 className="mt-1 text-2xl font-bold text-slate-900">
              {data.commit_short}{" "}
              <span className="font-mono text-base text-slate-500">
                on {data.branch ?? "—"}
              </span>
            </h1>
            <p className="mt-1 text-sm text-slate-600">{data.author_email}</p>
            {data.workflow_name && (
              <p className="mt-2 inline-flex items-center gap-2 rounded bg-slate-100 px-2 py-1 text-xs text-slate-700">
                <span className="uppercase tracking-wide text-slate-500">workflow</span>
                <span className="font-mono">{data.workflow_name}</span>
                {data.workflow_run_url && (
                  <a
                    href={data.workflow_run_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    open run ↗
                  </a>
                )}
              </p>
            )}
          </div>
          <div className="flex flex-col items-end gap-2">
            <RiskBadge decision={data.decision} score={data.risk_score} />
            <button
              onClick={() => setShowOverride(true)}
              className="rounded border border-slate-300 px-3 py-1 text-xs text-slate-700 hover:bg-slate-50"
            >
              Override…
            </button>
          </div>
        </div>
        {data.overridden_at && (
          <div className="mt-3 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
            <b>Overridden</b> at {new Date(data.overridden_at).toLocaleString()} —
            рішення було вручну змінено оператором.
          </div>
        )}
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatBox label="Predicted class" value={data.predicted_class} />
          <StatBox label="Confidence" value={`${(data.confidence * 100).toFixed(0)}%`} />
          <StatBox
            label="Memory (MB)"
            value={data.predicted_memory_mb ? data.predicted_memory_mb.toFixed(0) : "—"}
          />
          <StatBox
            label="Duration (min)"
            value={data.predicted_duration_min ? data.predicted_duration_min.toFixed(1) : "—"}
          />
        </div>
        {data.actual_outcome !== null && (
          <div className="mt-4 flex items-start gap-3 rounded border border-slate-200 bg-slate-50 p-3 text-sm">
            <div className="flex-1">
              <p className="font-medium text-slate-700">
                Actual outcome (heuristic):{" "}
                <span className="font-mono">{data.actual_outcome}</span>{" "}
                {data.actual_outcome === data.predicted_class ? (
                  <span className="ml-1 inline-flex items-center rounded bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800">
                    match
                  </span>
                ) : (
                  <span className="ml-1 inline-flex items-center rounded bg-rose-100 px-2 py-0.5 text-xs font-medium text-rose-800">
                    mismatch
                  </span>
                )}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Виведено евристикою з CI-логу (regex-патерни з failure_heuristics.py).
                Це шумна ground-truth — збіг означає узгодженість моделі з евристикою,
                не обов'язково з фактичною першопричиною збою.
              </p>
            </div>
          </div>
        )}
      </header>

      <section className="grid gap-6 md:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-700">
            Class probabilities
          </h2>
          <ul className="space-y-2">
            {sortedProbs.map(([cls, prob]) => (
              <li key={cls} className="flex items-center gap-3 text-sm">
                <ClassChip value={cls as never} />
                <div className="relative h-2 flex-1 rounded-full bg-slate-100">
                  <div
                    className="absolute inset-y-0 left-0 rounded-full bg-blue-500"
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-xs text-slate-600">
                  {(prob * 100).toFixed(1)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-4 flex items-baseline justify-between gap-2">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-700">
              SHAP explanation (top 10)
            </h2>
            <span
              className="text-[10px] uppercase tracking-wide text-slate-400"
              title="Локальні signed SHAP values за TreeExplainer. У two-stage-моделі пояснюється бінарна risk-модель (P(failure)) — та сама, що керує AUTO/WARN/BLOCK."
            >
              {data.shap_explanation
                ? `local · target=${data.shap_explanation.target}`
                : `legacy · |shap|`}
            </span>
          </div>
          {data.shap_explanation ? (
            <FeatureImportanceChart explanation={data.shap_explanation} />
          ) : (
            <FeatureImportanceChartLegacy importance={data.feature_importance} />
          )}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-700">
          Recommendations ({data.recommendations.length})
        </h2>
        <div className="space-y-3">
          {data.recommendations.map((rec, i) => (
            <RecommendationCard key={i} rec={rec} />
          ))}
          {data.recommendations.length === 0 && (
            <p className="rounded border border-slate-200 bg-white p-4 text-sm text-slate-500">
              Ризик низький — рекомендації не потрібні.
            </p>
          )}
        </div>
      </section>

      <footer className="text-xs text-slate-400">
        Inference time: {data.inference_time_ms} ms · Created: {new Date(data.created_at).toLocaleString()}
      </footer>

      {showOverride && (
        <OverrideModal
          predictionId={data.id}
          currentDecision={data.decision}
          onClose={() => setShowOverride(false)}
        />
      )}
    </div>
  );
}
