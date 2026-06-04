import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createPolicy,
  deletePolicy,
  fetchPolicies,
  Policy,
  PolicyInput,
  updatePolicy,
} from "../api";

const blankPolicy: PolicyInput = {
  name: "",
  auto_approve_threshold: 0.2,
  warn_threshold: 0.4,
  block_threshold: 0.6,
  allow_override: true,
  specific_rules: {},
  is_default: false,
};

function ThresholdInput({
  label,
  value,
  onChange,
  tone,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  tone: "emerald" | "amber" | "rose";
}) {
  const tones = {
    emerald: "border-emerald-300 focus:border-emerald-500",
    amber: "border-amber-300 focus:border-amber-500",
    rose: "border-rose-300 focus:border-rose-500",
  } as const;
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-xs uppercase tracking-wide text-slate-500">{label}</span>
      <input
        type="number"
        step="0.05"
        min={0}
        max={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`w-24 rounded border bg-white px-2 py-1 font-mono text-sm focus:outline-none ${tones[tone]}`}
      />
    </label>
  );
}

function PolicyForm({
  initial,
  onCancel,
  onSaved,
}: {
  initial: Policy | null;
  onCancel: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<PolicyInput>(
    initial
      ? {
          name: initial.name,
          auto_approve_threshold: initial.auto_approve_threshold,
          warn_threshold: initial.warn_threshold,
          block_threshold: initial.block_threshold,
          allow_override: initial.allow_override,
          specific_rules: initial.specific_rules,
          is_default: initial.is_default,
        }
      : blankPolicy,
  );
  const [error, setError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () =>
      initial ? updatePolicy(initial.id, form) : createPolicy(form),
    onSuccess: () => {
      setError(null);
      onSaved();
    },
    onError: (err: unknown) => {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(msg ?? "Save failed");
    },
  });

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        mutation.mutate();
      }}
      className="space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm"
    >
      <h2 className="text-base font-semibold text-slate-900">
        {initial ? "Редагування політики" : "Нова політика"}
      </h2>
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-xs uppercase tracking-wide text-slate-500">Назва</span>
        <input
          required
          maxLength={128}
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
          className="w-full rounded border border-slate-300 px-3 py-1.5 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <div className="grid grid-cols-3 gap-4">
        <ThresholdInput
          label="Auto approve <"
          value={form.auto_approve_threshold}
          onChange={(v) => setForm({ ...form, auto_approve_threshold: v })}
          tone="emerald"
        />
        <ThresholdInput
          label="Warn <"
          value={form.warn_threshold}
          onChange={(v) => setForm({ ...form, warn_threshold: v })}
          tone="amber"
        />
        <ThresholdInput
          label="Block ≥"
          value={form.block_threshold}
          onChange={(v) => setForm({ ...form, block_threshold: v })}
          tone="rose"
        />
      </div>
      <p className="text-xs text-slate-500">
        Логіка: risk_score &lt; auto_approve → AUTO_APPROVE; &lt; block → WARN; ≥ block → BLOCK.
        Пороги мають бути впорядковані: auto ≤ warn ≤ block.
      </p>
      <div className="flex items-center gap-6">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={form.allow_override}
            onChange={(e) => setForm({ ...form, allow_override: e.target.checked })}
          />
          Дозволити ручний override
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={form.is_default}
            onChange={(e) => setForm({ ...form, is_default: e.target.checked })}
          />
          Зробити політикою за замовчуванням
        </label>
      </div>
      {error && <p className="text-sm text-rose-600">{error}</p>}
      <div className="flex justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          className="rounded border border-slate-300 px-4 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
        >
          Скасувати
        </button>
        <button
          type="submit"
          disabled={mutation.isPending}
          className="rounded-md bg-blue-600 px-4 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-blue-700 disabled:opacity-50"
        >
          {mutation.isPending ? "Збереження..." : "Зберегти"}
        </button>
      </div>
    </form>
  );
}

export default function PoliciesSettings() {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["policies"], queryFn: fetchPolicies });
  const [editing, setEditing] = useState<Policy | null>(null);
  const [creating, setCreating] = useState(false);

  const delMutation = useMutation({
    mutationFn: deletePolicy,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["policies"] }),
  });

  const close = () => {
    setCreating(false);
    setEditing(null);
    qc.invalidateQueries({ queryKey: ["policies"] });
  };

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-8 py-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Політики</h1>
          <p className="mt-1 text-sm text-slate-500">
            Правила прийняття рішень за risk_score.
          </p>
        </div>
        <button
          onClick={() => {
            setCreating(true);
            setEditing(null);
          }}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-blue-700"
        >
          + Нова політика
        </button>
      </div>

      {creating && <PolicyForm initial={null} onCancel={close} onSaved={close} />}
      {editing && <PolicyForm initial={editing} onCancel={close} onSaved={close} />}

      {isLoading ? (
        <p className="text-slate-500">Завантаження...</p>
      ) : (
        <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-3">Назва</th>
                <th className="px-4 py-3 text-right">Auto &lt;</th>
                <th className="px-4 py-3 text-right">Warn &lt;</th>
                <th className="px-4 py-3 text-right">Block ≥</th>
                <th className="px-4 py-3">За замовч.</th>
                <th className="px-4 py-3 text-right">Дії</th>
              </tr>
            </thead>
            <tbody>
              {(data ?? []).map((p) => (
                <tr key={p.id} className="border-t border-slate-100">
                  <td className="px-4 py-3 font-medium text-slate-900">{p.name}</td>
                  <td className="px-4 py-3 text-right font-mono text-emerald-700">
                    {p.auto_approve_threshold.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-amber-700">
                    {p.warn_threshold.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-rose-700">
                    {p.block_threshold.toFixed(2)}
                  </td>
                  <td className="px-4 py-3">
                    {p.is_default ? (
                      <span className="rounded bg-slate-900 px-2 py-0.5 text-xs font-medium text-white">
                        default
                      </span>
                    ) : null}
                  </td>
                  <td className="space-x-2 px-4 py-3 text-right">
                    <button
                      onClick={() => {
                        setEditing(p);
                        setCreating(false);
                      }}
                      className="text-blue-600 hover:underline"
                    >
                      Редагувати
                    </button>
                    <button
                      disabled={p.is_default}
                      onClick={() => {
                        if (confirm(`Видалити політику "${p.name}"?`)) {
                          delMutation.mutate(p.id);
                        }
                      }}
                      className="text-rose-600 hover:underline disabled:cursor-not-allowed disabled:text-slate-400 disabled:no-underline"
                    >
                      Видалити
                    </button>
                  </td>
                </tr>
              ))}
              {!data?.length && (
                <tr>
                  <td colSpan={6} className="px-4 py-12 text-center text-sm text-slate-500">
                    Поки що немає політик. Створіть першу — її буде застосовано до всіх репозиторіїв.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
