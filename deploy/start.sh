#!/bin/bash
# Startup script — runs both API and Streamlit dashboard on one EC2 instance

echo "🚀 Starting Agentic Fraud Detection System..."

# Start FastAPI in background
echo "   Starting API on port 8000..."
cd /app
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready
echo "   Waiting for API..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✅ API ready"
        break
    fi
    sleep 1
done

# Start Streamlit
echo "   Starting Dashboard on port 8501..."
streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
DASH_PID=$!

echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ System Running"
echo "  API:       http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Dashboard: http://localhost:8501"
echo "═══════════════════════════════════════════════"
echo ""

# Wait for either process to exit
wait -n $API_PID $DASH_PID
echo "⚠️  A process exited. Shutting down..."
kill $API_PID $DASH_PID 2>/dev/null
