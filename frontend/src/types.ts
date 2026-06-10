export type RouteMode = "docs" | "web" | "both";

export type HealthResponse = {
  status: string;
  collection: string;
  model: string;
};

export type Source = {
  type: "docs" | "web";
  title: string;
  url: string;
  snippet: string;
  score: number;
};

export type SearchResponse = {
  query: string;
  answer: string;
  sources: Source[];
  route_used: Array<"docs" | "web">;
  model: string;
};

export type IngestResponse = {
  job_id: string;
  status: "queued";
  total_urls: number;
};

export type JobStatus = "queued" | "running" | "completed" | "failed";

export type JobState = {
  job_id: string;
  status: JobStatus;
  total_urls: number;
  pages_scraped: number;
  chunks_upserted: number;
  error?: string | null;
  created_at: string;
  updated_at: string;
};
