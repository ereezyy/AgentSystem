# Learnings from Predictive BI Optimization

## Blocking DB Operations in Async Functions
- Issue: Using  with synchronous  calls blocks the asyncio event loop.
- Solution: Convert  to  for functions performing blocking IO without . FastAPI/Starlette will execute these in a thread pool.
- Impact: Prevents blocking of concurrent requests, improving responsiveness.

## Verification
- Used reproduction scripts to demonstrate blocking behavior (2s delay).
- Verified fix by showing concurrent tasks proceed immediately (0.1s overhead).
- Verified syntax with .
- Verified importability with mocked dependencies.

## Cloud Agent environment

Start the Bottle server from `AgentSystem/` so `static/` is the document root:

```bash
cd AgentSystem
/opt/agentsystem-venv/bin/python main.py --mode server --host 0.0.0.0 --port 8000 --no-browser --no-email
```

- Pinned dependencies live in `AgentSystem/requirements.txt` and are installed into `/opt/agentsystem-venv`. FastAPI, Starlette, aiohttp, and pytest-asyncio are installed as well because `requirements-prod.txt` and the test suite import them.
- Redis listens on `127.0.0.1:6379` with no password. The task API does not need Postgres, Docker Compose, or Playwright browsers. Memory is SQLite under `./data/memory`.
- `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` must be non-empty at import. Boot supplies obvious placeholders when those variables are unset. Live model calls need real keys; startup logs a 401 for a placeholder and continues.
- Tests: `PYTHONPATH=/workspace /opt/agentsystem-venv/bin/pytest -c AgentSystem/pytest.ini AgentSystem/tests --continue-on-collection-errors`
- `GET /tasks` raises `AttributeError` because `AgentState` has no `get_all_tasks`. Create and read one task with `POST /task` and `GET /task/<id>`.
