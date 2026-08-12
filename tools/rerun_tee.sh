#!/usr/bin/env bash
# Live rerun web viewer AND a .rrd recording at the same time, on rerun 0.22.
#
# The Python SDK has a single sink before rerun 0.23, so the visualizer node can
# either serve the web viewer or write a file, never both. The rerun *CLI* has
# the same limit inside one process (`--serve-web --save` writes the file and
# never opens the viewer ports -- measured), but two CLI processes chained over
# the WebSocket stream do not:
#
#   visualizer node --TCP--> [proxy: --serve-web] --WS--> [saver: --save x.rrd]
#                                    |
#                                    +--WS--> browser
#
# Start this, then launch the visualizer pointed at the proxy's TCP port:
#
#   ros2 launch trajectory_following_ros2 mpc.launch.py ... \
#       viz_backend:=rerun viz_spawn_viewer:=false viz_serve_web:=false \
#       viz_connect_addr:=127.0.0.1:9876
#
# Order matters only for the recording: data logged before the saver attaches is
# buffered by the proxy and forwarded when it connects, but start the tee first
# and you never have to think about it.
#
# Usage:
#   tools/rerun_tee.sh start [out.rrd]   # default: /tmp/rerun_<timestamp>.rrd
#   tools/rerun_tee.sh status
#   tools/rerun_tee.sh stop              # flush + close the .rrd
#
# Ports (override with the matching env var): TCP_PORT 9876 (SDK in),
# WS_PORT 9877 (stream out), WEB_PORT 9090 (HTTP viewer).
set -euo pipefail

TCP_PORT=${TCP_PORT:-9876}
WS_PORT=${WS_PORT:-9877}
WEB_PORT=${WEB_PORT:-9090}
RUN_DIR=${RUN_DIR:-/tmp/rerun_tee}
PROXY_PID_FILE="$RUN_DIR/proxy.pid"
SAVER_PID_FILE="$RUN_DIR/saver.pid"

# Use the CLI bundled with the installed SDK, not whatever `rerun` is on PATH:
# the wire protocol is not compatible across rerun versions, and a mismatched
# CLI connects and then silently records nothing.
find_cli() {
    local cli
    cli=$(python3 - <<'PY' 2>/dev/null || true
import os
import rerun
print(os.path.join(os.path.dirname(os.path.dirname(rerun.__file__)), 'rerun_cli', 'rerun'))
PY
)
    if [[ -n "$cli" && -x "$cli" ]]; then
        echo "$cli"
        return 0
    fi
    echo "ERROR: no rerun CLI bundled with the installed rerun-sdk." >&2
    echo "       python3 -c 'import rerun; print(rerun.__version__)' to check the install." >&2
    return 1
}

lan_ip() {
    python3 -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('10.255.255.255', 1))
    print(s.getsockname()[0])
except OSError:
    print('localhost')
"
}

alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

start() {
    local out=${1:-/tmp/rerun_$(date +%Y%m%d_%H%M%S).rrd}
    local cli host
    cli=$(find_cli)
    mkdir -p "$RUN_DIR" "$(dirname "$out")"

    if alive "$PROXY_PID_FILE"; then
        echo "Already running (proxy pid $(cat "$PROXY_PID_FILE")). Run 'stop' first." >&2
        exit 1
    fi

    # A leftover process from an earlier run holds the ports, and the CLI's
    # response to that is a Rust panic backtrace that scrolls the useful output
    # away while a stale proxy keeps serving stale data.
    for p in "$TCP_PORT" "$WS_PORT" "$WEB_PORT"; do
        if ! python3 -c "
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('0.0.0.0', $p))
except OSError:
    sys.exit(1)
"; then
            echo "ERROR: port $p is already in use (leftover rerun? another run?)." >&2
            echo "       Free it, or re-run with TCP_PORT/WS_PORT/WEB_PORT set." >&2
            exit 1
        fi
    done

    nohup "$cli" --serve-web --bind 0.0.0.0 \
        --port "$TCP_PORT" --ws-server-port "$WS_PORT" --web-viewer-port "$WEB_PORT" \
        >"$RUN_DIR/proxy.log" 2>&1 &
    echo $! >"$PROXY_PID_FILE"

    # Wait for the WS port before attaching the saver; connecting too early just
    # fails and leaves a 12-byte header file that never grows.
    for _ in $(seq 40); do
        if python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(0.2)
sys.exit(0 if s.connect_ex(('127.0.0.1', $WS_PORT)) == 0 else 1)
"; then break; fi
        sleep 0.25
    done

    nohup "$cli" "ws://127.0.0.1:$WS_PORT" --save "$out" \
        >"$RUN_DIR/saver.log" 2>&1 &
    echo $! >"$SAVER_PID_FILE"
    echo "$out" >"$RUN_DIR/recording.path"

    sleep 1
    host=$(lan_ip)
    # Take the first line in bash, not through `| head -1`: head closes the pipe
    # early and the CLI panics on the broken pipe, printing a Rust backtrace over
    # the instructions below.
    local ver
    ver=$("$cli" --version 2>/dev/null)
    ver=${ver%%$'\n'*}
    cat <<EOF
rerun tee up ($ver)
  SDK connects to : 127.0.0.1:$TCP_PORT   (viz_connect_addr:=127.0.0.1:$TCP_PORT)
  recording       : $out
  web viewer      : http://localhost:$WEB_PORT/?url=ws://localhost:$WS_PORT
                    http://$host:$WEB_PORT/?url=ws://$host:$WS_PORT
The ?url= suffix is required -- a bare http://host:$WEB_PORT is the empty start page.
Use the second URL from any other machine. Stop with: tools/rerun_tee.sh stop
EOF
}

status() {
    local out='(none)'
    [[ -f "$RUN_DIR/recording.path" ]] && out=$(cat "$RUN_DIR/recording.path")
    alive "$PROXY_PID_FILE" && echo "proxy: running (pid $(cat "$PROXY_PID_FILE"))" || echo 'proxy: down'
    alive "$SAVER_PID_FILE" && echo "saver: running (pid $(cat "$SAVER_PID_FILE"))" || echo 'saver: down'
    echo "recording: $out"
    [[ -f "$out" ]] && ls -l "$out"
    return 0
}

stop() {
    # SIGINT first so the saver flushes and closes the .rrd, then escalate: the
    # CLI does not always exit on SIGINT when it was started detached, and a
    # leftover process holds the ports against the next run.
    for f in "$SAVER_PID_FILE" "$PROXY_PID_FILE"; do
        if alive "$f"; then
            local pid
            pid=$(cat "$f")
            kill -INT "$pid" 2>/dev/null || true
            for _ in $(seq 12); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.25
            done
            kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
            sleep 0.5
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        fi
        rm -f "$f"
    done
    if [[ -f "$RUN_DIR/recording.path" ]]; then
        local out cli
        out=$(cat "$RUN_DIR/recording.path")
        # The CLI saver writes one chunk per incoming message, where the SDK's own
        # rr.save() batches them -- measured 85.5 MB vs 10.1 MB for the same 19 s of
        # data, ~4.5 MB/s, which is gigabytes over a real run. Compacting recovers
        # all of it (85.5 -> 2.6 MB in 5 s) and the result opens identically.
        if [[ "${COMPACT:-1}" == 1 && -f "$out" ]]; then
            cli=$(find_cli)
            echo "Compacting $out ..."
            if "$cli" rrd compact --max-rows 4096 --max-bytes 1048576 \
                    "$out" -o "$out.compact" 2>/dev/null && [[ -s "$out.compact" ]]; then
                mv "$out.compact" "$out"
            else
                rm -f "$out.compact"
                echo "  (compaction failed; keeping the raw recording)" >&2
            fi
        fi
        [[ -f "$out" ]] && ls -l "$out"
        echo "Replay with: $(find_cli) $out"
    fi
}

case "${1:-start}" in
    start)  shift || true; start "${1:-}" ;;
    status) status ;;
    stop)   stop ;;
    *) echo "Usage: $0 {start [out.rrd]|status|stop}" >&2; exit 2 ;;
esac
