.PHONY: install bootstrap finalize backend frontend dev clean \
        docker-build docker-up docker-down docker-logs

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

bootstrap:
	python -m scripts.bootstrap_collection

# Run once after your initial big ingest completes.
finalize:
	python -m scripts.finalize_collection

backend:
	uvicorn backend.main:app --reload --host $${BACKEND_HOST:-0.0.0.0} --port $${BACKEND_PORT:-8000}

frontend:
	streamlit run frontend/app.py

# Run backend + frontend in parallel (requires GNU make)
dev:
	$(MAKE) -j2 backend frontend

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache

# --- Docker -------------------------------------------------------------
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
