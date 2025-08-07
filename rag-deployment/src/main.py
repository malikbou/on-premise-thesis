from fastapi import FastAPI

app = FastAPI(title="RAG API")

@app.get("/health")
def health_check():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok"}
