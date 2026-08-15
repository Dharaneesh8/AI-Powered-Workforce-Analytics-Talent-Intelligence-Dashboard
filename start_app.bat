@echo off
echo Starting AI Workforce Analytics Platform...

start "Backend Server (FastAPI)" cmd /k "cd /d "%~dp0ai workforce analysis\ai-workforce-analysis" && python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000"

start "Frontend (Vite + React)" cmd /k "cd /d "%~dp0ai workforce analysis\ai-workforce-analysis\frontend" && npm run dev"

echo Backend running on http://127.0.0.1:8000
echo Frontend running on http://localhost:5173
