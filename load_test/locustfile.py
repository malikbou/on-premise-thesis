"""Locust traffic generator for RAG Performance Tests.

Run locally (with server already running on http://localhost:8000):

    $ source .venv/bin/activate
    $ locust -f load_test/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to configure users, spawn rate, etc.

Environment variables:
    TESTSET_FILE – path to a JSON file with a list of {"user_input": ...}
                   default: testset/ragas_testset.json
"""

from __future__ import annotations

import json
import os
import random
from typing import List

from locust import HttpUser, task, between, events
import logging

# ---------------------------------------------------
# Load questions once at startup
# ---------------------------------------------------

def _load_questions() -> List[str]:
    dataset_path = os.getenv("TESTSET_FILE", "testset/ragas_testset.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Question dataset not found at {dataset_path}. Set TESTSET_FILE env var."
        )
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    return [record["user_input"] for record in data]


QUESTIONS = _load_questions()


# ---------------------------------------------------
# Locust user behaviour
# ---------------------------------------------------

class RAGUser(HttpUser):
    wait_time = between(0.5, 2.0)  # simulate think time between requests

    @task
    def ask_question(self):
        question = random.choice(QUESTIONS)
        payload = {"question": question}
        with self.client.post("/query", json=payload, catch_response=True) as resp:
            # Consider status >=500 as failure
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:100]}")
            else:
                resp.success()


# ---------------------------------------------------
# Event hooks (optional logging)
# ---------------------------------------------------

@events.test_start.add_listener  # type: ignore[attr-defined]
def on_test_start(environment, **_kwargs):
    logging.getLogger("locust").info(
        "Loaded %d questions from dataset – spawning users", len(QUESTIONS)
    )
