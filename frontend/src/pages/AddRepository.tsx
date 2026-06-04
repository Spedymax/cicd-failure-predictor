import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createRepository, fetchRepositories, Repository } from "../api";

function CopyableBlock({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs uppercase tracking-wide text-slate-500">
        <span>{label}</span>
        <button
          onClick={async () => {
            await navigator.clipboard.writeText(value);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
          className="rounded border border-slate-300 px-2 py-0.5 text-xs text-slate-700 hover:bg-slate-50"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto rounded border border-slate-200 bg-slate-50 p-2 font-mono text-xs text-slate-800">
        {value}
      </pre>
    </div>
  );
}

function WebhookInstructionsModal({
  repo,
  onClose,
}: {
  repo: Repository;
  onClose: () => void;
}) {
  const webhookUrl = `${window.location.origin}/api/v1/webhook/github`;
  const secret = repo.webhook_secret ?? "(missing)";
  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-slate-900/60 p-4">
      <div className="max-h-[90vh] w-full max-w-2xl space-y-5 overflow-y-auto rounded-lg bg-white p-6 shadow-xl">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-bold text-slate-900">Repository added</h2>
            <p className="text-sm text-slate-600">
              <span className="font-mono">{repo.full_name}</span> створено. Тепер налаштуйте webhook у GitHub:
            </p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            ✕
          </button>
        </div>
        <CopyableBlock label="Payload URL" value={webhookUrl} />
        <CopyableBlock label="Webhook secret (показано один раз!)" value={secret} />
        <div className="rounded border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          <p className="font-semibold">Інструкція:</p>
          <ol className="ml-5 mt-2 list-decimal space-y-1">
            <li>
              Відкрийте репозиторій на GitHub →{" "}
              <span className="font-mono">Settings → Webhooks → Add webhook</span>
            </li>
            <li>Вставте <b>Payload URL</b> та <b>Secret</b> з полів вище.</li>
            <li>Content type: <span className="font-mono">application/json</span></li>
            <li>
              Events: <span className="font-mono">push</span>, <span className="font-mono">workflow_run</span>
            </li>
            <li>Save webhook.</li>
          </ol>
          <p className="mt-2 text-xs">
            Перший вхідний webhook завершить налаштування — мова та dockerfile-флаги
            визначаться автоматично.
          </p>
        </div>
        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700"
          >
            Готово
          </button>
        </div>
      </div>
    </div>
  );
}

export default function AddRepository() {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ["repositories"],
    queryFn: fetchRepositories,
  });
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<Repository | null>(null);

  const mutation = useMutation({
    mutationFn: () => createRepository({ full_name: fullName }),
    onSuccess: (repo) => {
      setError(null);
      setCreated(repo);
      setFullName("");
      qc.invalidateQueries({ queryKey: ["repositories"] });
    },
    onError: (err: unknown) => {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(msg ?? "Failed to add repository");
    },
  });

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-8 py-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Repositories</h1>
        <p className="mt-1 text-sm text-slate-500">
          Підключені репозиторії та підказка для налаштування webhook у GitHub.
        </p>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (fullName.includes("/")) mutation.mutate();
          else setError("Format must be owner/repo");
        }}
        className="space-y-3 rounded-lg border border-slate-200 bg-white p-6 shadow-sm"
      >
        <h2 className="text-base font-semibold text-slate-900">Add a repository</h2>
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs uppercase tracking-wide text-slate-500">
            owner/repo (на GitHub)
          </span>
          <input
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            placeholder="myorg/my-service"
            className="w-full rounded border border-slate-300 px-3 py-2 font-mono focus:border-blue-500 focus:outline-none"
          />
        </label>
        {error && <p className="text-sm text-rose-600">{error}</p>}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={!fullName || mutation.isPending}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 disabled:opacity-50"
          >
            {mutation.isPending ? "Adding..." : "Add repository"}
          </button>
        </div>
      </form>

      <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-3">Repository</th>
              <th className="px-4 py-3">Branch</th>
              <th className="px-4 py-3">Language</th>
              <th className="px-4 py-3">Dockerfile</th>
              <th className="px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-slate-500">
                  Завантаження...
                </td>
              </tr>
            )}
            {(data ?? []).map((r) => (
              <tr key={r.id} className="border-t border-slate-100">
                <td className="px-4 py-3 font-mono text-slate-900">
                  <a href={r.url} target="_blank" rel="noreferrer" className="hover:underline">
                    {r.full_name}
                  </a>
                </td>
                <td className="px-4 py-3 font-mono text-xs text-slate-600">{r.default_branch}</td>
                <td className="px-4 py-3 text-slate-700">{r.language ?? "—"}</td>
                <td className="px-4 py-3">{r.has_dockerfile ? "Yes" : "—"}</td>
                <td className="px-4 py-3">
                  {r.is_active ? (
                    <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800">
                      active
                    </span>
                  ) : (
                    <span className="rounded bg-slate-200 px-2 py-0.5 text-xs font-medium text-slate-700">
                      inactive
                    </span>
                  )}
                </td>
              </tr>
            ))}
            {!isLoading && !data?.length && (
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-sm text-slate-500">
                  Поки що немає репозиторіїв. Додайте перший формою вище.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {created && <WebhookInstructionsModal repo={created} onClose={() => setCreated(null)} />}
    </div>
  );
}
