import { NavLink } from "react-router-dom";
import type { ComponentType, SVGProps } from "react";
import { useAuth } from "../auth/AuthContext";

type IconProps = SVGProps<SVGSVGElement>;

const IconStroke = (path: string) => (props: IconProps) => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <path d={path} />
  </svg>
);

const ListIcon = IconStroke(
  "M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01",
);
const ChartIcon = IconStroke(
  "M3 3v18h18M7 14l3-3 4 4 5-6",
);
const ShieldIcon = IconStroke(
  "M12 3l8 3v6c0 5-3.5 8.5-8 9-4.5-.5-8-4-8-9V6l8-3zM9 12l2 2 4-4",
);
const FolderIcon = IconStroke(
  "M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V7z",
);

const NAV: {
  to: string;
  label: string;
  Icon: ComponentType<IconProps>;
  end?: boolean;
}[] = [
  { to: "/", label: "Прогнози", Icon: ListIcon, end: true },
  { to: "/analytics", label: "Аналітика", Icon: ChartIcon },
  { to: "/policies", label: "Політики", Icon: ShieldIcon },
  { to: "/repositories", label: "Репозиторії", Icon: FolderIcon },
];

export function Sidebar() {
  const { user, logout } = useAuth();
  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-slate-200 bg-slate-50">
      <div className="flex items-center gap-3 border-b border-slate-200 px-5 py-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-md bg-blue-600 text-sm font-bold text-white shadow-sm">
          CI
        </div>
        <div className="leading-tight">
          <p className="text-sm font-semibold text-slate-900">Predictor</p>
          <p className="text-[10px] uppercase tracking-wider text-slate-500">
            v26
          </p>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 p-3">
        {NAV.map((it) => (
          <NavLink
            key={it.to}
            to={it.to}
            end={it.end}
            className={({ isActive }) =>
              `group flex items-center gap-3 rounded-md px-3 py-2 text-sm transition ${
                isActive
                  ? "bg-white text-blue-700 shadow-sm ring-1 ring-blue-100"
                  : "text-slate-600 hover:bg-white hover:text-slate-900"
              }`
            }
          >
            {({ isActive }) => (
              <>
                <it.Icon
                  className={
                    isActive ? "text-blue-600" : "text-slate-400 group-hover:text-slate-600"
                  }
                />
                <span className="font-medium">{it.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="space-y-2 border-t border-slate-200 p-4 text-xs text-slate-500">
        {user && (
          <div className="space-y-1 rounded-md bg-white px-3 py-2 ring-1 ring-slate-200">
            <p className="truncate text-sm font-medium text-slate-800">
              {user.name || user.email}
            </p>
            <p className="truncate text-[10px] uppercase tracking-wider text-slate-400">
              {user.role}
            </p>
            <button
              type="button"
              onClick={logout}
              className="mt-1 w-full rounded-md border border-slate-200 px-2 py-1 text-[11px] font-medium text-slate-600 transition hover:bg-slate-100 hover:text-slate-900"
            >
              Вийти
            </button>
          </div>
        )}
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60"></span>
            <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500"></span>
          </span>
          <span>Бекенд активний</span>
        </div>
        <p className="text-[10px] text-slate-400">
          Дипломна робота · ІП-з21 · 2026
        </p>
      </div>
    </aside>
  );
}
