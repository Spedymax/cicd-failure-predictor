import type { FailureClass } from "../api";

const COLORS: Record<FailureClass, string> = {
  success: "bg-emerald-50 text-emerald-700 border-emerald-200",
  oom_killed: "bg-rose-50 text-rose-700 border-rose-200",
  test_timeout: "bg-amber-50 text-amber-700 border-amber-200",
  test_failure: "bg-orange-50 text-orange-700 border-orange-200",
  dependency_error: "bg-blue-50 text-blue-700 border-blue-200",
  docker_build_failed: "bg-purple-50 text-purple-700 border-purple-200",
  network_error: "bg-cyan-50 text-cyan-700 border-cyan-200",
  other_failure: "bg-slate-100 text-slate-700 border-slate-300",
};

export function ClassChip({ value }: { value: FailureClass }) {
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 font-mono text-xs ${COLORS[value]}`}
    >
      {value}
    </span>
  );
}
