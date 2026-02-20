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
