import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  LinearScale,
  Tooltip,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import type { ShapExplanation } from "../api";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

// Legacy fallback: flat {feature: abs_share} dict — used for predictions
// stored before signed SHAP was introduced.
export function FeatureImportanceChartLegacy({
  importance,
}: {
  importance: Record<string, number>;
}) {
  const entries = Object.entries(importance).slice(0, 8);
  const labels = entries.map(([name]) => name.replace(/^feat_/, ""));
  const values = entries.map(([, score]) => score);
  return (
    <Bar
      data={{
        labels,
        datasets: [
          {
            label: "|SHAP| (normalised)",
            data: values,
            backgroundColor: "#3b82f6",
            borderRadius: 4,
          },
        ],
      }}
      options={{
        indexAxis: "y",
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { callback: (v) => `${(Number(v) * 100).toFixed(0)}%` } },
        },
      }}
    />
  );
}

function fmtValue(v: number): string {
  if (Number.isInteger(v)) return v.toString();
  if (Math.abs(v) < 0.01) return v.toExponential(1);
  return v.toFixed(2);
}

export function FeatureImportanceChart({
  explanation,
}: {
  explanation: ShapExplanation;
}) {
  const top = explanation.contributions.slice(0, 10);
  const labels = top.map(
    (c) => `${c.feature.replace(/^feat_/, "")} = ${fmtValue(c.value)}`,
  );
  const values = top.map((c) => c.shap_value);
  // Positive shap → pushes failure prob up → red; negative → green.
  const colors = values.map((v) => (v >= 0 ? "#ef4444" : "#10b981"));
  const targetLabel =
    explanation.target === "risk_failure"
      ? "P(failure)"
      : explanation.target.replace(/^class:/, "P(") + ")";
  return (
    <div className="space-y-2">
      <Bar
        data={{
          labels,
          datasets: [
            {
              label: `SHAP → ${targetLabel}`,
              data: values,
              backgroundColor: colors,
              borderRadius: 4,
            },
          ],
        }}
        options={{
          indexAxis: "y",
          responsive: true,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const v = ctx.parsed.x as number;
                  const sign = v >= 0 ? "+" : "";
                  return `Δ ${targetLabel}: ${sign}${v.toFixed(4)}`;
                },
              },
            },
          },
          scales: {
            x: {
              grid: { color: (c) => (c.tick.value === 0 ? "#475569" : "#e2e8f0") },
              ticks: { callback: (v) => Number(v).toFixed(2) },
            },
          },
        }}
      />
      <p className="text-[11px] text-slate-500">
        base = {explanation.base_value.toFixed(3)} → predicted ={" "}
        {explanation.predicted_value.toFixed(3)}. Червоні стовпці підвищують{" "}
        {targetLabel}, зелені — знижують.
      </p>
    </div>
  );
}
