import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Title,
  Tooltip,
} from "chart.js";
import { Bar, Doughnut } from "react-chartjs-2";
import { fetchTrends, TrendsResponse } from "../api";

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend);

const WINDOWS = [7, 30, 90] as const;
type Window = (typeof WINDOWS)[number];

const CLASS_COLORS: Record<string, string> = {
  success: "#10b981",
  oom_killed: "#dc2626",
  test_timeout: "#f59e0b",
  dependency_error: "#8b5cf6",
  docker_build_failed: "#0ea5e9",
  network_error: "#ec4899",
  other_failure: "#64748b",
};

function StatTile({ label, value, tone }: { label: string; value: string | number; tone?: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-1 text-2xl font-semibold ${tone ?? "text-slate-900"}`}>{value}</p>
    </div>
  );
}

function DailyStackedBar({ data }: { data: TrendsResponse["daily"] }) {
  const labels = data.map((d) => d.date);
  const config = {
    labels,
    datasets: [
      { label: "AUTO_APPROVE", data: data.map((d) => d.auto_approve), backgroundColor: "#10b981" },
      { label: "WARN", data: data.map((d) => d.warn), backgroundColor: "#f59e0b" },
      { label: "BLOCK", data: data.map((d) => d.block), backgroundColor: "#dc2626" },
    ],
  };
  return (
    <Bar
      data={config}
      options={{
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
        plugins: { legend: { position: "bottom" } },
      }}
    />
  );
}

function FailureClassDoughnut({ data }: { data: Record<string, number> }) {
  const labels = Object.keys(data);
  const values = labels.map((k) => data[k]);
  const colors = labels.map((k) => CLASS_COLORS[k] ?? "#94a3b8");
  return (
    <Doughnut
      data={{ labels, datasets: [{ data: values, backgroundColor: colors }] }}
      options={{
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "right" } },
      }}
    />
  );
}

function TopReposBar({ data }: { data: TrendsResponse["top_repos"] }) {
  if (!data.length) return <p className="text-sm text-slate-500">Замало даних для агрегації.</p>;
  return (
    <Bar
      data={{
        labels: data.map((r) => r.repo),
        datasets: [
          {
            label: "Середній risk_score",
            data: data.map((r) => r.avg_risk),
            backgroundColor: "#0ea5e9",
          },
        ],
      }}
      options={{
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y" as const,
        scales: { x: { beginAtZero: true, max: 1.0 } },
        plugins: { legend: { display: false } },
      }}
    />
  );
}

export default function Analytics() {
  const [days, setDays] = useState<Window>(30);
  const { data, isLoading, error } = useQuery({
    queryKey: ["trends", days],
    queryFn: () => fetchTrends(days),
  });

  if (isLoading) return <p className="p-6 text-slate-500">Завантаження аналітики...</p>;
  if (error || !data) return <p className="p-6 text-rose-600">Не вдалося завантажити статистику</p>;

  const blockShare = data.totals.n_predictions
    ? Math.round((data.totals.n_block / data.totals.n_predictions) * 100)
    : 0;

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-8 py-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Аналітика</h1>
          <p className="mt-1 text-sm text-slate-500">
            Тренди прогнозів за вибране вікно часу.
          </p>
        </div>
        <div className="inline-flex rounded-md border border-slate-200 bg-white p-0.5 text-sm shadow-sm">
          {WINDOWS.map((w) => (
            <button
              key={w}
              onClick={() => setDays(w)}
              className={`rounded px-3 py-1 transition ${
                days === w
                  ? "bg-blue-600 text-white shadow-sm"
                  : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              {w} дн
            </button>
          ))}
        </div>
      </div>

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile label="Усього прогнозів" value={data.totals.n_predictions} />
        <StatTile label="AUTO_APPROVE" value={data.totals.n_auto} tone="text-emerald-600" />
        <StatTile label="WARN" value={data.totals.n_warn} tone="text-amber-600" />
        <StatTile label="BLOCK" value={`${data.totals.n_block} (${blockShare}%)`} tone="text-rose-600" />
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-700">
          Рішення за день
        </h2>
        <div className="h-64">
          <DailyStackedBar data={data.daily} />
        </div>
      </section>

      <section className="grid gap-6 md:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-700">
            Розподіл прогнозованих класів збою
          </h2>
          <div className="h-64">
            <FailureClassDoughnut data={data.failure_class} />
          </div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-700">
            Топ-10 репозиторіїв за середнім risk
          </h2>
          <div className="h-64">
            <TopReposBar data={data.top_repos} />
          </div>
        </div>
      </section>

      <footer className="text-xs text-slate-400">
        Вікно: {data.window_days} днів · Від {new Date(data.since).toLocaleString()}
      </footer>
    </div>
  );
}
