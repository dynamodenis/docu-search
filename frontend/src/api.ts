import type {
  HealthResponse,
  IngestResponse,
  JobState,
  RouteMode,
  SearchResponse,
} from "./types";

const DEFAULT_BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

export function getStoredBackendUrl() {
  return localStorage.getItem("docu-search.backend-url") || DEFAULT_BACKEND_URL;
}

export function storeBackendUrl(url: string) {
  localStorage.setItem("docu-search.backend-url", url.replace(/\/$/, ""));
}

async function request<T>(
  backendUrl: string,
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${backendUrl.replace(/\/$/, "")}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Keep the HTTP status fallback when the body is not JSON.
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export function getHealth(backendUrl: string) {
  return request<HealthResponse>(backendUrl, "/health", {
    method: "GET",
  });
}

export function searchDocs(
  backendUrl: string,
  payload: {
    query: string;
    top_k: number;
    model?: string;
    force_route?: RouteMode;
  },
) {
  return request<SearchResponse>(backendUrl, "/search", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function startIngest(
  backendUrl: string,
  payload: {
    urls: string[];
    sitemap_url?: string;
    source_label: string;
    max_pages: number;
  },
) {
  return request<IngestResponse>(backendUrl, "/ingest", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getJob(backendUrl: string, jobId: string) {
  return request<JobState>(backendUrl, `/jobs/${jobId}`, {
    method: "GET",
  });
}
