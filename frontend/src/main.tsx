import React, { FormEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import {
  Activity,
  BookOpen,
  CheckCircle2,
  Clock3,
  Globe2,
  Database,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Server,
  Settings2,
  UploadCloud,
  XCircle,
} from "lucide-react";
import {
  getHealth,
  getJob,
  getStoredBackendUrl,
  searchDocs,
  startIngest,
  storeBackendUrl,
} from "./api";
import type { HealthResponse, JobState, RouteMode, SearchResponse, Source } from "./types";
import "./styles.css";

const routeOptions: Array<{ label: string; value: "auto" | RouteMode }> = [
  { label: "Auto", value: "auto" },
  { label: "Docs (Qdrant)", value: "docs" },
  { label: "Web (Tavily)", value: "web" },
  { label: "Docs + Web", value: "both" },
];

function App() {
  const [backendUrl, setBackendUrl] = useState(getStoredBackendUrl);
  const [draftBackendUrl, setDraftBackendUrl] = useState(backendUrl);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState("");
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);

  const [query, setQuery] = useState("");
  const [model, setModel] = useState("");
  const [route, setRoute] = useState<"auto" | RouteMode>("auto");
  const [topK, setTopK] = useState(5);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [searchError, setSearchError] = useState("");
  const [isSearching, setIsSearching] = useState(false);

  const [urls, setUrls] = useState("");
  const [sitemapUrl, setSitemapUrl] = useState("");
  const [sourceLabel, setSourceLabel] = useState("user_submitted");
  const [maxPages, setMaxPages] = useState(500);
  const [ingestError, setIngestError] = useState("");
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState<JobState | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [activeView, setActiveView] = useState<"chat" | "ingest">("chat");
  const [activeSourceTab, setActiveSourceTab] = useState<"docs" | "web">("docs");

  async function refreshHealth(url = backendUrl) {
    setIsCheckingHealth(true);
    setHealthError("");
    try {
      const result = await getHealth(url);
      setHealth(result);
    } catch (error) {
      setHealth(null);
      setHealthError(error instanceof Error ? error.message : "Backend is unreachable.");
    } finally {
      setIsCheckingHealth(false);
    }
  }

  useEffect(() => {
    void refreshHealth();
  }, [backendUrl]);

  useEffect(() => {
    if (!jobId) return;

    let cancelled = false;
    let timer: number | undefined;

    async function poll(delay = 2000) {
      try {
        const nextJob = await getJob(backendUrl, jobId);
        if (cancelled) return;
        setJob(nextJob);
        if (nextJob.status === "completed" || nextJob.status === "failed") return;
        timer = window.setTimeout(() => void poll(Math.min(delay + 1000, 8000)), delay);
      } catch (error) {
        if (!cancelled) {
          setIngestError(error instanceof Error ? error.message : "Could not read job status.");
        }
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [backendUrl, jobId]);

  const groupedSources = useMemo(() => {
    const docs = searchResult?.sources.filter((source) => source.type === "docs") ?? [];
    const web = searchResult?.sources.filter((source) => source.type === "web") ?? [];
    return { docs, web };
  }, [searchResult]);

  function saveBackendUrl(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextUrl = draftBackendUrl.trim().replace(/\/$/, "");
    if (!nextUrl) return;
    storeBackendUrl(nextUrl);
    setBackendUrl(nextUrl);
  }

  async function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    setSearchError("");
    setSearchResult(null);
    try {
      const result = await searchDocs(backendUrl, {
        query: query.trim(),
        top_k: topK,
        model: model.trim() || undefined,
        force_route: route === "auto" ? undefined : route,
      });
      setSearchResult(result);
    } catch (error) {
      setSearchError(error instanceof Error ? error.message : "Search failed.");
    } finally {
      setIsSearching(false);
    }
  }

  async function submitIngest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const urlList = urls
      .split("\n")
      .map((url) => url.trim())
      .filter(Boolean);

    if (!urlList.length && !sitemapUrl.trim()) {
      setIngestError("Provide URLs or a sitemap URL.");
      return;
    }

    setIsIngesting(true);
    setIngestError("");
    setJob(null);
    try {
      const response = await startIngest(backendUrl, {
        urls: urlList,
        sitemap_url: sitemapUrl.trim() || undefined,
        source_label: sourceLabel.trim() || "user_submitted",
        max_pages: maxPages,
      });
      setJobId(response.job_id);
    } catch (error) {
      setIngestError(error instanceof Error ? error.message : "Ingest failed.");
    } finally {
      setIsIngesting(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <Search size={22} />
          </span>
          <div>
            <h1>docu-search</h1>
            <p>Hybrid docs and web RAG</p>
          </div>
        </div>

        <section className="panel compact">
          <div className="panel-title">
            <Server size={18} />
            <span>Backend</span>
            <button
              className="icon-button"
              type="button"
              onClick={() => void refreshHealth()}
              aria-label="Refresh backend status"
              title="Refresh backend status"
            >
              <RefreshCw size={16} className={isCheckingHealth ? "spin" : ""} />
            </button>
          </div>

          <form className="backend-form" onSubmit={saveBackendUrl}>
            <input
              value={draftBackendUrl}
              onChange={(event) => setDraftBackendUrl(event.target.value)}
              aria-label="Backend URL"
            />
            <button type="submit">Save</button>
          </form>

          {health ? (
            <div className="status good">
              <CheckCircle2 size={17} />
              <span>{health.status}</span>
            </div>
          ) : (
            <div className="status bad">
              <XCircle size={17} />
              <span>{healthError || "Not checked"}</span>
            </div>
          )}

          {health && (
            <dl className="meta-list">
              <div>
                <dt>Collection</dt>
                <dd>{health.collection}</dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{health.model}</dd>
              </div>
            </dl>
          )}
        </section>

        <section className="panel compact">
          <div className="panel-title">
            <Settings2 size={18} />
            <span>Search Options</span>
          </div>
          <label>
            <span>Model override</span>
            <input
              value={model}
              onChange={(event) => setModel(event.target.value)}
              placeholder="openai/gpt-4o-mini"
            />
          </label>
          <label>
            <span>Top K sources</span>
            <input
              type="range"
              min="1"
              max="20"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
            <strong>{topK}</strong>
          </label>
          <div className="field-heading">Retrieval route</div>
          <div className="segmented" aria-label="Retrieval route">
            {routeOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                className={route === option.value ? "active" : ""}
                onClick={() => setRoute(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </section>

        <div className="powered-by">
          <Database size={16} />
          <span>Powered by</span>
          <strong>Qdrant</strong>
        </div>
      </aside>

      <section className="workspace">
        <nav className="view-tabs" aria-label="Workspace views">
          <button
            type="button"
            className={activeView === "chat" ? "active" : ""}
            onClick={() => setActiveView("chat")}
          >
            <Search size={17} />
            Chat
          </button>
          <button
            type="button"
            className={activeView === "ingest" ? "active" : ""}
            onClick={() => setActiveView("ingest")}
          >
            <UploadCloud size={17} />
            Ingest
            {jobId && <span className="tab-dot" />}
          </button>
        </nav>

        {activeView === "chat" && (
          <>
            <form className="search-bar" onSubmit={submitSearch}>
              <Search size={22} />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Ask anything about your indexed docs..."
                autoFocus
              />
              <button type="submit" disabled={isSearching || !query.trim()}>
                {isSearching ? <Loader2 size={18} className="spin" /> : <Play size={18} />}
                Search
              </button>
            </form>

            {searchError && <div className="banner error">{searchError}</div>}

            <section className="answer-layout">
              <article className="answer-panel">
                {!searchResult && !isSearching && (
                  <div className="empty-state">
                    <BookOpen size={40} />
                    <h2>Ask your documentation a question</h2>
                    <p>
                      The React frontend calls the same FastAPI endpoints as the old Streamlit
                      app, but it stays awake as a normal browser app.
                    </p>
                  </div>
                )}

                {isSearching && (
                  <div className="loading-state">
                    <Loader2 size={34} className="spin" />
                    <h2>Routing, retrieving, generating...</h2>
                  </div>
                )}

                {searchResult && (
                  <>
                    <div className="answer-header">
                      <div>
                        <span className="eyebrow">Answer</span>
                        <h2>{searchResult.query}</h2>
                      </div>
                      <div className="route-badges">
                        {(searchResult.route_used.length ? searchResult.route_used : ["none"]).map(
                          (item) => (
                            <span key={item}>{item}</span>
                          ),
                        )}
                      </div>
                    </div>
                    <div className="markdown">
                      <ReactMarkdown>{searchResult.answer}</ReactMarkdown>
                    </div>
                    <p className="model-line">Generated with {searchResult.model}</p>
                  </>
                )}
              </article>

              <aside className="sources-panel">
                <div className="source-tabs" aria-label="Source types">
                  <button
                    type="button"
                    className={activeSourceTab === "docs" ? "active" : ""}
                    onClick={() => setActiveSourceTab("docs")}
                  >
                    <BookOpen size={16} />
                    Documentation
                    <span>{groupedSources.docs.length}</span>
                  </button>
                  <button
                    type="button"
                    className={activeSourceTab === "web" ? "active" : ""}
                    onClick={() => setActiveSourceTab("web")}
                  >
                    <Globe2 size={16} />
                    Web
                    <span>{groupedSources.web.length}</span>
                  </button>
                </div>

                {activeSourceTab === "docs" ? (
                  <SourceGroup
                    title="Documentation"
                    icon={<BookOpen size={18} />}
                    sources={groupedSources.docs}
                  />
                ) : (
                  <SourceGroup title="Web" icon={<Globe2 size={18} />} sources={groupedSources.web} />
                )}
              </aside>
            </section>
          </>
        )}

        {activeView === "ingest" && (
          <section className="ingest-section">
            <div className="section-heading">
              <UploadCloud size={21} />
              <div>
                <h2>Ingest sources</h2>
                <p>Queue pages or a sitemap for scraping, chunking, embedding, and upsert.</p>
              </div>
            </div>

            <form className="ingest-grid" onSubmit={submitIngest}>
              <label className="span-2">
                <span>URLs</span>
                <textarea
                  value={urls}
                  onChange={(event) => setUrls(event.target.value)}
                  placeholder="https://example.com/docs/getting-started&#10;https://example.com/docs/config"
                />
              </label>
              <label>
                <span>Sitemap URL</span>
                <input
                  value={sitemapUrl}
                  onChange={(event) => setSitemapUrl(event.target.value)}
                  placeholder="https://example.com/sitemap.xml"
                />
              </label>
              <label>
                <span>Source label</span>
                <input
                  value={sourceLabel}
                  onChange={(event) => setSourceLabel(event.target.value)}
                />
              </label>
              <label>
                <span>Max pages</span>
                <input
                  type="number"
                  min="1"
                  max="10000"
                  value={maxPages}
                  onChange={(event) => setMaxPages(Number(event.target.value))}
                />
              </label>
              <button className="primary span-2" type="submit" disabled={isIngesting}>
                {isIngesting ? <Loader2 size={18} className="spin" /> : <UploadCloud size={18} />}
                Start ingestion
              </button>
            </form>

            {ingestError && <div className="banner error">{ingestError}</div>}
            {jobId && <JobCard jobId={jobId} job={job} />}
          </section>
        )}
      </section>
    </main>
  );
}

function SourceGroup({
  title,
  icon,
  sources,
}: {
  title: string;
  icon: React.ReactNode;
  sources: Source[];
}) {
  return (
    <section className="source-group">
      <div className="source-heading">
        {icon}
        <h3>{title}</h3>
        <span>{sources.length}</span>
      </div>
      {sources.length === 0 ? (
        <p className="muted">No sources yet.</p>
      ) : (
        sources.map((source, index) => (
          <article className="source-card" key={`${source.type}-${source.url}-${index}`}>
            <div className="source-title">
              <strong>{source.title || source.url || "Untitled source"}</strong>
              <span>{source.score.toFixed(2)}</span>
            </div>
            {source.url && (
              <a href={source.url} target="_blank" rel="noreferrer">
                {source.url}
              </a>
            )}
            {source.snippet && <p>{source.snippet}</p>}
          </article>
        ))
      )}
    </section>
  );
}

function JobCard({ jobId, job }: { jobId: string; job: JobState | null }) {
  const done = job?.status === "completed";
  const failed = job?.status === "failed";
  const running = job?.status === "queued" || job?.status === "running";

  return (
    <article className="job-card">
      <div className="job-title">
        {done && <CheckCircle2 size={19} />}
        {failed && <XCircle size={19} />}
        {running && <Activity size={19} />}
        {!job && <Clock3 size={19} />}
        <strong>{job?.status ?? "queued"}</strong>
        <code>{jobId}</code>
      </div>
      <div className="job-metrics">
        <span>Pages scraped: {job?.pages_scraped ?? 0}</span>
        <span>Chunks upserted: {job?.chunks_upserted ?? 0}</span>
        <span>Total URLs: {job?.total_urls ?? "..."}</span>
      </div>
      {job?.error && <p className="job-error">{job.error}</p>}
    </article>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
