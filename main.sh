
#!/usr/bin/env bash

set -Eeuo pipefail

# Project directory
PROJECT_DIR="/home/alireza/Alireza/project/aptar_company_project/code/v1"

# Conda installation
CONDA_BASE="$HOME/anaconda3"

# llama.cpp PID file
PID_FILE="$PROJECT_DIR/run/llama-server.pid"

# Stop llama.cpp when the application exits
cleanup() {
    echo ""
    echo "Closing Persian RAG..."

    if [[ -f "$PID_FILE" ]]; then
        LLAMA_PID=$(cat "$PID_FILE")

        if kill -0 "$LLAMA_PID" 2>/dev/null; then
            echo "Stopping llama.cpp server..."
            kill -TERM "$LLAMA_PID" 2>/dev/null || true

            # Wait briefly for the server to stop
            for attempt in {1..10}; do
                if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done

            # Force stop only if it is still running
            if kill -0 "$LLAMA_PID" 2>/dev/null; then
                kill -KILL "$LLAMA_PID" 2>/dev/null || true
            fi
        fi

        rm -f "$PID_FILE"
    fi

    echo "Persian RAG closed."
}

# Cleanup on normal exit or terminal interruption
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Activate Conda
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py312

# Enter project directory
cd "$PROJECT_DIR"

# Create runtime directory
mkdir -p "$PROJECT_DIR/run"

# Start Streamlit
echo "Starting Persian RAG..."

python -m streamlit run main_v2.py
