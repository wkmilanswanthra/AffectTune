pnpm --filter web dev &
uvicorn services.api-gateway.main:app --reload --port 8000
