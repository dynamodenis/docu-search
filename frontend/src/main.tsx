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

const REPO_URL = "https://github.com/dynamodenis/docu-search";

// lucide deprecated its brand icons, so inline the official GitHub mark.
function GithubIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.44 9.8 8.21 11.39.6.11.82-.26.82-.58 0-.29-.01-1.05-.02-2.06-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.21.09 1.85 1.24 1.85 1.24 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.34-5.47-5.95 0-1.32.47-2.39 1.24-3.23-.13-.31-.54-1.53.11-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6.01 0c2.29-1.55 3.3-1.23 3.3-1.23.65 1.65.24 2.87.12 3.18.77.84 1.23 1.91 1.23 3.23 0 4.62-2.81 5.64-5.49 5.94.43.37.81 1.1.81 2.22 0 1.61-.01 2.9-.01 3.29 0 .32.21.7.82.58A12.01 12.01 0 0 0 24 12.5C24 5.87 18.63.5 12 .5z" />
    </svg>
  );
}
import {
  getHealth,
  getJob,
  getSources,
  getStoredBackendUrl,
  searchDocs,
  startIngest,
  storeBackendUrl,
} from "./api";
import type {
  HealthResponse,
  JobState,
  RouteMode,
  SearchResponse,
  Source,
  SourceLabel,
} from "./types";
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
  const [activeView, setActiveView] = useState<"chat" | "ingest" | "sources">("chat");
  const [activeSourceTab, setActiveSourceTab] = useState<"docs" | "web">("docs");

  const [dataSources, setDataSources] = useState<SourceLabel[]>([]);
  const [sourcesError, setSourcesError] = useState("");
  const [isLoadingSources, setIsLoadingSources] = useState(false);
  const [sourceFilter, setSourceFilter] = useState<string | null>(null);

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

  async function refreshSources(url = backendUrl) {
    setIsLoadingSources(true);
    setSourcesError("");
    try {
      const result = await getSources(url);
      setDataSources(result.sources);
    } catch (error) {
      setDataSources([]);
      setSourcesError(error instanceof Error ? error.message : "Could not load sources.");
    } finally {
      setIsLoadingSources(false);
    }
  }

  useEffect(() => {
    void refreshHealth();
    void refreshSources();
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
        if (nextJob.status === "completed" || nextJob.status === "failed") {
          // New chunks just landed in Qdrant — refresh the data-source list
          // so the newly-ingested label (and its count) shows up.
          if (nextJob.status === "completed") void refreshSources();
          return;
        }
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
        source_label: sourceFilter || undefined,
      });
      setSearchResult(result);
    } catch (error) {
      setSearchError(error instanceof Error ? error.message : "Search failed.");
    } finally {
      setIsSearching(false);
    }
  }

  function filterBySource(label: string) {
    // Scope subsequent searches to this ingested source and jump to chat.
    setSourceFilter(label);
    setActiveView("chat");
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
          <label className="pt-10">
            <span>Top K sources</span>
            <div className="range-row">
              <input
                type="range"
                min="1"
                max="20"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
              <strong className="range-value">{topK}</strong>
            </div>
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
      </aside>

      <section className="workspace">
        <header className="workspace-top">
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
          <button
            type="button"
            className={activeView === "sources" ? "active" : ""}
            onClick={() => setActiveView("sources")}
          >
            <Database size={17} />
            Data sources
            {dataSources.length > 0 && <span className="tab-count">{dataSources.length}</span>}
          </button>
          </nav>

          <div className="top-actions">
            <a
              className="github-link"
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              aria-label="View source on GitHub"
              title="View source on GitHub"
            >
              <GithubIcon size={18} />
            </a>
            <a
              className="powered-by"
              href="https://qdrant.tech"
              target="_blank"
              rel="noreferrer"
            >
              <Database size={16} />
              <span>Powered by</span>
              <strong>Qdrant</strong>
            </a>
          </div>
        </header>

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

            {sourceFilter && (
              <div className="filter-chip">
                <Database size={15} />
                <span>
                  Scoped to <strong>{sourceFilter}</strong>
                </span>
                <button
                  type="button"
                  onClick={() => setSourceFilter(null)}
                  aria-label="Clear source filter"
                  title="Clear source filter"
                >
                  <XCircle size={15} />
                </button>
              </div>
            )}

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

        {activeView === "sources" && (
          <section className="ingest-section">
            <div className="section-heading">
              <Database size={21} />
              <div>
                <h2>Data sources</h2>
                <p>Everything currently indexed in Qdrant, grouped by the label it was ingested under.</p>
              </div>
              <button
                className="icon-button light"
                type="button"
                onClick={() => void refreshSources()}
                aria-label="Refresh data sources"
                title="Refresh data sources"
              >
                <RefreshCw size={16} className={isLoadingSources ? "spin" : ""} />
              </button>
            </div>

            {sourcesError && <div className="banner error">{sourcesError}</div>}

            {!sourcesError && dataSources.length === 0 ? (
              <div className="empty-state">
                <Database size={40} />
                <h2>{isLoadingSources ? "Loading sources..." : "No sources indexed yet"}</h2>
                <p>Ingest some docs and they'll show up here, grouped by source label.</p>
              </div>
            ) : (
              <div className="source-list">
                {dataSources.map((source) => {
                  const active = sourceFilter === source.label;
                  return (
                    <button
                      type="button"
                      className={`source-row${active ? " active" : ""}`}
                      key={source.label}
                      onClick={() => filterBySource(source.label)}
                      title={`Search only ${source.label}`}
                    >
                      <span className="source-badge">
                        <Database size={16} />
                      </span>
                      <div className="source-meta">
                        <strong>{source.label}</strong>
                        <span>{source.chunks.toLocaleString()} chunks</span>
                      </div>
                      <span className="source-action">
                        {active ? "Filtering" : "Search this"}
                        <Search size={14} />
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
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
