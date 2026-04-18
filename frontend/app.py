"""Streamlit UI. Calls the FastAPI backend via api_client."""
from __future__ import annotations

import os
import time

import streamlit as st
from dotenv import load_dotenv

from frontend.api_client import APIClient

load_dotenv()

st.set_page_config(
    page_title="docu-search",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def get_client() -> APIClient:
    return APIClient(base_url=os.getenv("BACKEND_URL"))


api = get_client()

st.title("docu-search")
st.caption(
    "Hybrid RAG — Qdrant (dense + BM25 + ColBERT) plus Tavily web search, "
    "routed by an LLM via tool-use. Built by Denis Mbugua."
)

# Sidebar: status + global options
with st.sidebar:
    st.subheader("Status")
    try:
        h = api.health()
        st.success(f"backend {h['status']}")
        st.caption(f"collection: `{h['collection']}`")
        st.caption(f"model: `{h['model']}`")
    except Exception as e:  # noqa: BLE001
        st.error(f"backend unreachable: {e}")

    st.divider()
    st.subheader("Options")
    model_override = st.text_input(
        "OpenRouter model (optional)",
        placeholder="e.g. openai/gpt-4o-mini",
        help="Leave empty to use the backend default.",
    )
    force_route = st.selectbox(
        "Force route",
        options=["auto (LLM decides)", "docs only", "web only", "docs + web"],
        index=0,
    )
    top_k = st.slider("Top K sources per tool call", 1, 10, 5)

route_map = {
    "auto (LLM decides)": None,
    "docs only": "docs",
    "web only": "web",
    "docs + web": "both",
}

search_tab, ingest_tab = st.tabs(["Search", "Ingest"])

# ------------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------------
with search_tab:
    query = st.text_input(
        "Ask anything",
        placeholder="how do I configure HNSW for better recall?",
    )

    if query:
        with st.spinner("Routing, retrieving, generating..."):
            try:
                result = api.search(
                    query=query,
                    top_k=top_k,
                    model=model_override or None,
                    force_route=route_map[force_route],
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Search failed: {e}")
                st.stop()

        route_used = result.get("route_used") or []
        st.caption(
            f"Routed via: **{', '.join(route_used) or 'none'}** • "
            f"model: `{result.get('model', '?')}`"
        )

        st.subheader("Answer")
        st.markdown(result["answer"])

        sources = result.get("sources", [])
        if sources:
            doc_sources = [s for s in sources if s["type"] == "docs"]
            web_sources = [s for s in sources if s["type"] == "web"]

            if doc_sources:
                st.subheader("Documentation sources")
                for i, s in enumerate(doc_sources, 1):
                    with st.expander(f"[{i}] {s['title']}  (score {s['score']:.2f})"):
                        if s.get("url"):
                            st.markdown(f"[{s['url']}]({s['url']})")
                        st.caption(s.get("snippet", ""))

            if web_sources:
                st.subheader("Web sources")
                for i, s in enumerate(web_sources, 1):
                    with st.expander(f"[{i}] {s['title']}  (score {s['score']:.2f})"):
                        if s.get("url"):
                            st.markdown(f"[{s['url']}]({s['url']})")
                        st.caption(s.get("snippet", ""))

# ------------------------------------------------------------------
# INGEST
# ------------------------------------------------------------------
with ingest_tab:
    st.markdown(
        "Add your own pages or a sitemap. Scraping, chunking, embedding, "
        "and upsert run in the background — this page polls for progress."
    )

    with st.form("ingest_form"):
        urls_raw = st.text_area(
            "URLs (one per line)",
            placeholder="https://example.com/docs/getting-started\nhttps://example.com/docs/config",
            height=140,
        )
        sitemap_url = st.text_input(
            "...or a sitemap URL",
            placeholder="https://example.com/sitemap.xml",
        )
        col1, col2 = st.columns(2)
        with col1:
            source_label = st.text_input("Source label", value="user_submitted")
        with col2:
            max_pages = st.number_input(
                "Max pages", min_value=1, max_value=500, value=50, step=5
            )
        submitted = st.form_submit_button("Start ingestion")

    if submitted:
        urls = [u.strip() for u in urls_raw.splitlines() if u.strip()]
        if not urls and not sitemap_url.strip():
            st.warning("Provide URLs or a sitemap URL.")
        else:
            try:
                resp = api.ingest(
                    urls=urls,
                    sitemap_url=sitemap_url.strip() or None,
                    source_label=source_label.strip() or "user_submitted",
                    max_pages=int(max_pages),
                )
                st.session_state["job_id"] = resp["job_id"]
                st.success(f"Queued job `{resp['job_id']}`")
            except Exception as e:  # noqa: BLE001
                st.error(f"Ingest failed: {e}")

    # Poll status of the current job.
    if st.session_state.get("job_id"):
        job_id = st.session_state["job_id"]
        placeholder = st.empty()
        for _ in range(300):  # ~10 min at 2s poll
            try:
                job = api.job(job_id)
            except Exception as e:  # noqa: BLE001
                placeholder.error(f"Could not read job status: {e}")
                break

            with placeholder.container():
                st.subheader("Current job")
                st.json(
                    {
                        "job_id": job["job_id"],
                        "status": job["status"],
                        "pages_scraped": job["pages_scraped"],
                        "chunks_upserted": job["chunks_upserted"],
                        "error": job.get("error"),
                    }
                )
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(2)
