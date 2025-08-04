### RAG Performance-Test Harness

This folder contains everything needed to **measure throughput, latency, and concurrency** of your Retrieval-Augmented Generation (RAG) application on your local machine.

Why keep it here?
1. **Shared code** – test server re-uses retriever & generation logic from `src/` without publishing a package.
2. **Clear separation** – accuracy benchmarking stays in `src/`, performance assets live here.
3. **Single clone reproducibility** – one repository, one virtual-env, one `docker-compose up`.

Components to be added (tracked via TODO list):
- `rag_server.py` – FastAPI wrapper around your RAG chain, instrumented with Prometheus.
- `locustfile.py` – traffic generator describing user load patterns.
- `prometheus.yml` – scrape config; Grafana dashboards JSON.
- `docker-compose.yml` – launches rag_server, Locust, Prometheus, and Grafana containers (arm64).

## Local usage (once scaffold is complete)
```bash
# Activate project virtual environment
source .venv/bin/activate

# Build & run the performance rig
cd load_test
export RAG_PORT=8000  # optional override
docker-compose up --build
```

Grafana will be available at http://localhost:3000 (admin / admin by default). Locust Web UI lives at http://localhost:8089.

> **Note:** All Docker images will be pinned to `linux/arm64` so they run on Apple Silicon.

---

Feel free to update this file with additional notes or troubleshooting tips as the harness evolves.
