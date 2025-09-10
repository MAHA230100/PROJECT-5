#!/usr/bin/env bash
set -euo pipefail

# Defaults for Streamlit
export STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-8501}
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

streamlit run app.py --server.port "$STREAMLIT_SERVER_PORT" --server.headless "$STREAMLIT_SERVER_HEADLESS"

