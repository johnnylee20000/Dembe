#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/workspace"
SESSION_NAME="streamlit-tt-legal"
TMUX_CFG="/exec-daemon/tmux.portal.conf"
APP_URL="http://localhost:8501"

cd "$APP_DIR"

# Ensure a tmux session exists for the app process.
if ! tmux -f "$TMUX_CFG" has-session -t "=$SESSION_NAME" 2>/dev/null; then
  tmux -f "$TMUX_CFG" new-session -d -s "$SESSION_NAME" -c "$APP_DIR" -- "${SHELL:-bash}" -l
fi

# Start Streamlit only if not already running in the session.
if ! tmux -f "$TMUX_CFG" capture-pane -t "$SESSION_NAME:0.0" -p | rg -q "streamlit run app_streamlit.py"; then
  tmux -f "$TMUX_CFG" send-keys -t "$SESSION_NAME:0.0" 'cd /workspace && streamlit run app_streamlit.py --server.address 0.0.0.0 --server.port 8501' C-m
fi

# Give server a brief moment before opening browser.
sleep 1
xdg-open "$APP_URL" >/dev/null 2>&1 || true
