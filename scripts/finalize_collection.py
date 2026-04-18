"""Switch the collection from bulk-ingest mode to search mode.

Run this ONCE after your initial big ingest finishes. Sets dense HNSW
to m=16, turns on indexing (threshold=1000), and waits for the
collection status to go green before returning.

Usage:
    python -m scripts.finalize_collection
"""
import logging

from backend.core.qdrant_client import finalize_indexing

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


if __name__ == "__main__":
    finalize_indexing(wait=True)
    print("done")
