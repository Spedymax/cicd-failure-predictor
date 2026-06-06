import axios from "axios";

const TOKEN_KEY = "predictor_jwt";

export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY);
export const setToken = (t: string): void => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = (): void => localStorage.removeItem(TOKEN_KEY);

const client = axios.create({
  baseURL: "/api/v1",
  timeout: 10_000,
});

client.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers = config.headers ?? {};
    (config.headers as Record<string, string>).Authorization = `Bearer ${token}`;
  }
  return config;
});

client.interceptors.response.use(
  (resp) => resp,
  (error) => {
    if (error?.response?.status === 401) {
      clearToken();
      if (!window.location.pathname.startsWith("/login")) {
        window.location.assign("/login");
      }
    }
    return Promise.reject(error);
  },
);

// ---------- Auth ----------

export type UserRole = "developer" | "devops" | "team_lead" | "admin";

export interface CurrentUser {
  id: number;
  email: string;
  name: string | null;
  role: UserRole;
  is_active: boolean;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export const login = async (email: string, password: string): Promise<CurrentUser> => {
  const { data } = await client.post<LoginResponse>("/auth/login", { email, password });
  setToken(data.access_token);
  return await fetchCurrentUser();
};

export const fetchCurrentUser = async (): Promise<CurrentUser> => {
  const { data } = await client.get<CurrentUser>("/auth/me");
  return data;
};

export const logout = (): void => {
  clearToken();
  window.location.assign("/login");
};

export type FailureClass =
  | "success"
  | "oom_killed"
  | "test_timeout"
  | "test_failure"
  | "dependency_error"
  | "docker_build_failed"
  | "network_error"
  | "other_failure";

export type Decision = "auto_approve" | "warn" | "block";

export interface ShapContribution {
  feature: string;
  value: number;
  shap_value: number;
}

export interface ShapExplanation {
  target: string;
  base_value: number;
  predicted_value: number;
  contributions: ShapContribution[];
}

export interface PredictionListItem {
  id: number;
  repository_full_name: string;
  commit_short: string;
  author_email: string;
  predicted_class: FailureClass;
  decision: Decision;
  risk_score: number;
  confidence: number;
  created_at: string;
}

export interface Recommendation {
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  category: string;
  title: string;
  description: string;
  actions: string[];
  estimated_impact: string | null;
}

export interface PredictionDetail {
  id: number;
  repository_id: number;
  repository_full_name: string;
  commit_sha: string;
  commit_short: string;
  author_email: string;
  branch: string | null;
  workflow_name: string | null;
  workflow_run_url: string | null;
  predicted_class: FailureClass;
  decision: Decision;
  risk_score: number;
  confidence: number;
  class_probabilities: Record<string, number>;
  feature_importance: Record<string, number>;
  shap_explanation: ShapExplanation | null;
  predicted_memory_mb: number | null;
  predicted_duration_min: number | null;
  recommendations: Recommendation[];
  inference_time_ms: number;
  overridden_at: string | null;
  actual_outcome: FailureClass | null;
  created_at: string;
}

export interface PredictionListResponse {
  items: PredictionListItem[];
  total: number;
  limit: number;
  offset: number;
}

export type SourceFilter = "all" | "demo" | "real";

export const fetchPredictions = async (
  limit = 50,
  source: SourceFilter = "all",
  predictedClass: FailureClass | null = null,
): Promise<PredictionListResponse> => {
  const params: Record<string, string | number> = { limit, source };
  if (predictedClass) params.predicted_class = predictedClass;
  const { data } = await client.get<PredictionListResponse>("/predictions", {
    params,
  });
  return data;
};

export const fetchPrediction = async (id: number): Promise<PredictionDetail> => {
  const { data } = await client.get<PredictionDetail>(`/predictions/${id}`);
  return data;
};

export const exportPredictions = async (
  format: "json" | "csv",
  source: SourceFilter = "all",
  predictedClass: FailureClass | null = null,
): Promise<void> => {
  const params: Record<string, string> = { format, source };
  if (predictedClass) params.predicted_class = predictedClass;
  const resp = await client.get("/predictions/export", {
    params,
    responseType: "blob",
  });
  const url = URL.createObjectURL(resp.data as Blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `predictions.${format}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};

// ---------- Override ----------

export const overridePrediction = async (
  id: number,
  newDecision: Decision,
  reason: string,
): Promise<PredictionDetail> => {
  const { data } = await client.post<PredictionDetail>(
    `/predictions/${id}/override`,
    { new_decision: newDecision, reason },
  );
  return data;
};

// ---------- Analytics ----------

export interface DailyStat {
  date: string;
  total: number;
  auto_approve: number;
  warn: number;
  block: number;
}
export interface TopRepoStat {
  repo: string;
  n: number;
  avg_risk: number;
}
export interface TrendsResponse {
  window_days: number;
  since: string;
  daily: DailyStat[];
  failure_class: Record<string, number>;
  top_repos: TopRepoStat[];
  totals: {
    n_predictions: number;
    n_block: number;
    n_warn: number;
    n_auto: number;
  };
}

export const fetchTrends = async (
  days = 30,
  source: SourceFilter = "all",
): Promise<TrendsResponse> => {
  const { data } = await client.get<TrendsResponse>("/stats/trends", {
    params: { days, source },
  });
  return data;
};

// ---------- Policies ----------

export interface Policy {
  id: number;
  name: string;
  auto_approve_threshold: number;
  warn_threshold: number;
  block_threshold: number;
  allow_override: boolean;
  specific_rules: Record<string, unknown>;
  is_default: boolean;
}

export type PolicyInput = Omit<Policy, "id">;

export const fetchPolicies = async (): Promise<Policy[]> => {
  const { data } = await client.get<Policy[]>("/policies");
  return data;
};
export const createPolicy = async (body: PolicyInput): Promise<Policy> => {
  const { data } = await client.post<Policy>("/policies", body);
  return data;
};
export const updatePolicy = async (id: number, body: PolicyInput): Promise<Policy> => {
  const { data } = await client.put<Policy>(`/policies/${id}`, body);
  return data;
};
export const deletePolicy = async (id: number): Promise<void> => {
  await client.delete(`/policies/${id}`);
};

// ---------- Repositories ----------

export interface Repository {
  id: number;
  provider: string;
  full_name: string;
  url: string;
  default_branch: string;
  ci_platform: string;
  language: string | null;
  package_manager: string | null;
  has_dockerfile: boolean;
  webhook_secret: string | null;  // populated only on create()
  policy_id: number | null;
  last_synced_at: string | null;
  is_active: boolean;
}

export interface RepositoryInput {
  full_name: string;
  default_branch?: string;
  policy_id?: number | null;
}

export const fetchRepositories = async (): Promise<Repository[]> => {
  const { data } = await client.get<Repository[]>("/repositories");
  return data;
};
export const createRepository = async (body: RepositoryInput): Promise<Repository> => {
  const { data } = await client.post<Repository>("/repositories", body);
  return data;
};
