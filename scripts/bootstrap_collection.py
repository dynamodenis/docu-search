"""One-time (idempotent) collection creation.

Usage:
    python -m scripts.bootstrap_collection
"""
import logging

from backend.core.qdrant_client import ensure_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


if __name__ == "__main__":
    created = ensure_collection()
    print("created" if created else "already existed")
