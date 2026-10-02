#!/usr/bin/env bash
# Start the medfm Lab UI.
#
#   ./run.sh              # foreground, Ctrl-C to stop
#   ./run.sh --detach     # background, survives closing the terminal
#   ./run.sh --port 7861  # different port
#
# Running it here rather than from an agent session is deliberate: a server started by an
# agent is a child of that agent's lifecycle and gets cleaned up when the session closes.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${MEDFM_PORT:-7860}"
DETACH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --port)   PORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x .venv/bin/python ]]; then
  echo "No virtualenv found. Create it first:" >&2
  echo "  uv venv --python 3.12 .venv" >&2
  echo "  VIRTUAL_ENV=\$PWD/.venv uv pip install -r requirements.txt" >&2
  exit 1
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use:" >&2
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2
  exit 1
fi

export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MEDFM_PORT="$PORT"

mkdir -p logs

if [[ "$DETACH" == "1" ]]; then
  # nohup + a redirect detach it from this shell, so closing the terminal does not kill it.
  # (macOS has no setsid.)
  nohup .venv/bin/python app.py > logs/server.log 2>&1 < /dev/null &
  PID=$!
  echo "started on http://127.0.0.1:$PORT (pid $PID)"
  echo "logs: $(pwd)/logs/server.log"
  sleep 2
  if kill -0 "$PID" 2>/dev/null; then
    echo "stop: kill $PID"
  else
    echo "process exited immediately — see logs/server.log" >&2
    tail -20 logs/server.log >&2
    exit 1
  fi
else
  trap 'echo; echo "stopping..."; exit 0' INT TERM
  exec .venv/bin/python app.py
fi
