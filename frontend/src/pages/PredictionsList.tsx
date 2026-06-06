import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import {
  exportPredictions,
  fetchPredictions,
  fetchTrends,
  type FailureClass,
  type SourceFilter,
} from "../api";
import { ClassChip } from "../components/ClassChip";
import { RiskBadge } from "../components/RiskBadge";
import { StatCard } from "../components/StatCard";

const TABS: { id: SourceFilter; label: string; hint: string }[] = [
  { id: "demo", label: "Демо репо", hint: "Синтетичні прогнози для перегляду UI" },
  { id: "real", label: "Реальні дані", hint: "Прогнози на справжніх GitHub Actions runs" },
];

const CLASSES: { id: FailureClass | "all"; label: string }[] = [
  { id: "all", label: "Усі класи" },
  { id: "success", label: "success" },
  { id: "oom_killed", label: "oom_killed" },
  { id: "test_timeout", label: "test_timeout" },
  { id: "test_failure", label: "test_failure" },
  { id: "dependency_error", label: "dependency_error" },
  { id: "docker_build_failed", label: "docker_build_failed" },
  { id: "network_error", label: "network_error" },
  { id: "other_failure", label: "other_failure" },
];

function shortenAuthor(email: string, max = 24): string {
  if (email.length <= max) return email;
  return email.slice(0, max - 1) + "…";
}

function timeAgo(iso: string): string {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return `${Math.floor(diff)} с тому`;
  if (diff < 3600) return `${Math.floor(diff / 60)} хв тому`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} год тому`;
  return `${Math.floor(diff / 86400)} дн тому`;
}

const WINDOWS: { days: number; label: string }[] = [
  { days: 7, label: "7 днів" },
  { days: 30, label: "30 днів" },
  { days: 90, label: "90 днів" },
];

export default function PredictionsList() {
  const [tab, setTab] = useState<SourceFilter>("demo");
  const [classFilter, setClassFilter] = useState<FailureClass | "all">("all");
  const [windowDays, setWindowDays] = useState<number>(7);
  const [exporting, setExporting] = useState<"json" | "csv" | null>(null);

  const handleExport = async (format: "json" | "csv") => {
    setExporting(format);
    try {
      await exportPredictions(format, tab, classFilter === "all" ? null : classFilter);
    } catch (err) {
      console.error("export failed", err);
    } finally {
      setExporting(null);
    }
  };

  const { data, isLoading, error } = useQuery({
    queryKey: ["predictions", tab, classFilter],
    queryFn: () =>
      fetchPredictions(50, tab, classFilter === "all" ? null : classFilter),
    refetchInterval: 5_000,
  });

  const { data: trends } = useQuery({
    queryKey: ["trends", windowDays, tab],
    queryFn: () => fetchTrends(windowDays, tab),
    refetchInterval: 15_000,
  });

  const total = trends?.totals.n_predictions ?? 0;
  const nAuto = trends?.totals.n_auto ?? 0;
  const nWarn = trends?.totals.n_warn ?? 0;
  const nBlock = trends?.totals.n_block ?? 0;
  const safe = (n: number) => (total > 0 ? n / total : 0);

  const githubUrl = (repo: string, sha: string) =>
    `https://github.com/${repo}/commit/${sha}`;
  const isRealRepo = (repo: string) =>
    !repo.startsWith("acme/") && !repo.startsWith("demo/");

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-8 py-8">
      <header className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
            Прогнози CI/CD
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            {data
              ? `Усього в базі ${data.total} прогнозів · показано ${data.items.length}`
              : "Завантаження..."}
          </p>
        </div>
        <div className="flex flex-col items-end gap-2 text-xs text-slate-400">
          <div className="inline-flex rounded-md border border-slate-200 bg-white p-0.5 shadow-sm">
            {WINDOWS.map((w) => (
              <button
                key={w.days}
                type="button"
                onClick={() => setWindowDays(w.days)}
                className={`rounded px-2.5 py-1 text-xs font-medium transition ${
                  windowDays === w.days
                    ? "bg-blue-600 text-white"
                    : "text-slate-500 hover:bg-slate-100 hover:text-slate-700"
                }`}
              >
                {w.label}
              </button>
            ))}
          </div>
          <p className="font-mono">оновлення ~5 с</p>
        </div>
      </header>

      <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Усього" value={total} hint={`${windowDays}-денне вікно`} tone="info" />
        <StatCard
          label="Auto approve"
          value={nAuto}
          tone="success"
          share={safe(nAuto)}
        />
        <StatCard
          label="Warn"
          value={nWarn}
          tone="warning"
          share={safe(nWarn)}
        />
        <StatCard
          label="Block"
          value={nBlock}
          tone="danger"
          share={safe(nBlock)}
        />
      </section>

      <section className="rounded-lg border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-5 py-3">
          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                title={t.hint}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
                  tab === t.id
                    ? "bg-blue-50 text-blue-700 ring-1 ring-blue-200"
                    : "text-slate-500 hover:bg-slate-100 hover:text-slate-700"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <div className="flex items-center gap-2 text-sm">
            <label htmlFor="class-filter" className="text-slate-500">
              Клас:
            </label>
            <select
              id="class-filter"
              value={classFilter}
              onChange={(e) => setClassFilter(e.target.value as FailureClass | "all")}
              className="rounded-md border border-slate-300 bg-white px-2.5 py-1.5 font-mono text-xs focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100"
            >
              {CLASSES.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
            <span className="mx-1 h-5 w-px bg-slate-200" aria-hidden />
            <span className="text-xs text-slate-400">Експорт:</span>
            <div className="inline-flex overflow-hidden rounded-md border border-slate-300 shadow-sm">
              <button
                type="button"
                onClick={() => handleExport("json")}
                disabled={exporting !== null}
                className="border-r border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {exporting === "json" ? "…" : "JSON"}
              </button>
              <button
                type="button"
                onClick={() => handleExport("csv")}
                disabled={exporting !== null}
                className="bg-white px-2.5 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {exporting === "csv" ? "…" : "CSV"}
              </button>
            </div>
          </div>
        </div>

        {isLoading && (
          <p className="px-6 py-12 text-center text-sm text-slate-500">Завантаження...</p>
        )}
        {error && (
          <p className="px-6 py-12 text-center text-sm text-rose-600">
            Помилка: {String(error)}
          </p>
        )}

        {data && (
          <div className="overflow-x-auto">
            <table className="min-w-full table-fixed divide-y divide-slate-200 text-sm">
              <colgroup>
                <col className="w-16" />
                <col className="w-56" />
                <col className="w-24" />
                <col className="w-48" />
                <col className="w-48" />
                <col className="w-36" />
                <col className="w-24" />
              </colgroup>
              <thead className="bg-slate-50 text-left text-[11px] font-medium uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="px-5 py-3">ID</th>
                  <th className="px-5 py-3">Репозиторій</th>
                  <th className="px-5 py-3">Commit</th>
                  <th className="px-5 py-3">Автор</th>
                  <th className="px-5 py-3">Клас</th>
                  <th className="px-5 py-3">Рішення</th>
                  <th className="px-5 py-3 text-right">Створено</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {data.items.map((p) => (
                  <tr key={p.id} className="transition hover:bg-slate-50">
                    <td className="px-5 py-3 font-mono text-xs text-slate-400">
                      #{p.id}
                    </td>
                    <td className="px-5 py-3 font-medium">
                      <Link
                        to={`/predictions/${p.id}`}
                        className="block truncate text-slate-900 hover:text-blue-700"
                        title={p.repository_full_name}
                      >
                        {p.repository_full_name}
                      </Link>
                    </td>
                    <td className="px-5 py-3 font-mono text-xs text-slate-600">
                      {isRealRepo(p.repository_full_name) ? (
                        <a
                          href={githubUrl(p.repository_full_name, p.commit_short)}
                          target="_blank"
                          rel="noreferrer"
                          className="hover:text-blue-700 hover:underline"
                        >
                          {p.commit_short}
                        </a>
                      ) : (
                        p.commit_short
                      )}
                    </td>
                    <td
                      className="truncate px-5 py-3 text-slate-600"
                      title={p.author_email}
                    >
                      {shortenAuthor(p.author_email)}
                    </td>
                    <td className="px-5 py-3">
                      <ClassChip value={p.predicted_class} />
                    </td>
                    <td className="px-5 py-3">
                      <RiskBadge decision={p.decision} score={p.risk_score} />
                    </td>
                    <td
                      className="px-5 py-3 text-right text-xs text-slate-500"
                      title={new Date(p.created_at).toLocaleString()}
                    >
                      {timeAgo(p.created_at)}
                    </td>
                  </tr>
                ))}
                {data.items.length === 0 && (
                  <tr>
                    <td
                      colSpan={7}
                      className="px-5 py-16 text-center text-sm text-slate-500"
                    >
                      У цій вкладці немає прогнозів за обраним фільтром.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
