#!/usr/bin/env bash
# Use Render's $PORT or default to 8000 for local testing
PORT=${PORT:-8000}
echo "Starting uvicorn on 0.0.0.0:${PORT}"
uvicorn main:app --host 0.0.0.0 --port "$PORT" --proxy-headers
