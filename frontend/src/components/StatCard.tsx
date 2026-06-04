import type { ReactNode } from "react";

type Tone = "default" | "success" | "warning" | "danger" | "info";

const TONES: Record<Tone, { value: string; bar: string; pill: string }> = {
  default: { value: "text-slate-900", bar: "bg-slate-200", pill: "bg-slate-100 text-slate-600" },
  success: { value: "text-emerald-700", bar: "bg-emerald-500", pill: "bg-emerald-50 text-emerald-700" },
  warning: { value: "text-amber-700", bar: "bg-amber-500", pill: "bg-amber-50 text-amber-700" },
  danger: { value: "text-rose-700", bar: "bg-rose-500", pill: "bg-rose-50 text-rose-700" },
  info: { value: "text-blue-700", bar: "bg-blue-500", pill: "bg-blue-50 text-blue-700" },
};

export function StatCard({
  label,
  value,
  hint,
  tone = "default",
  share,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: Tone;
  share?: number;
}) {
  const t = TONES[tone];
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-baseline justify-between">
        <p className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
          {label}
        </p>
        {typeof share === "number" && (
          <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium ${t.pill}`}>
            {(share * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <p className={`text-2xl font-semibold tabular-nums ${t.value}`}>{value}</p>
      {hint && <p className="text-xs text-slate-500">{hint}</p>}
      {typeof share === "number" && (
        <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className={`h-full ${t.bar}`}
            style={{ width: `${Math.min(100, Math.max(0, share * 100))}%` }}
          />
        </div>
      )}
    </div>
  );
}
