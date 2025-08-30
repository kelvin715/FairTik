#!/bin/bash

echo "Starting TikTok Revenue Allocation AI Assistant + Multimodal Live Stream Evaluation API Server..."

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found"
    exit 1
fi

# Check if model files exist
MODEL_PATH="/root/tiktok_techjam/policy_learning/final_dpo_model"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model files do not exist: $MODEL_PATH"
    echo "Please ensure DPO model has been trained and saved in the correct location"
    exit 1
fi

# Install dependencies (if needed)
echo "Checking dependencies..."
pip install -r requirements.txt

# Start server
echo "Starting API server at http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"
echo "Health check: http://localhost:8000/health"
echo ""
echo "Available endpoints:"
echo "- POST /api/revenue-allocation - Revenue allocation recommendation"
echo "- POST /api/live-stream-evaluation - Live stream content evaluation"
echo "- POST /api/batch-revenue-allocation - Batch revenue allocation recommendation"
echo "- POST /api/batch-live-stream-evaluation - Batch live stream evaluation"
echo ""
echo "Press Ctrl+C to stop server"

python3 api_server.py
