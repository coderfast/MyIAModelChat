#!/bin/bash
# Start the FastAPI server for MyIAModelChat
# Usage: bash runserver.sh [port]
PORT=${1:-11434}
echo "Starting envAIModels server on port $PORT..."
uvicorn envAIModels.app:app --reload --port "$PORT"
