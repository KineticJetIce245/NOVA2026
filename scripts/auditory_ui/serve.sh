#!/usr/bin/env sh
# The no-hardware auditory demo, for macOS and Linux: replay a converted KU Leuven
# trial through the fitted decoder and serve the dashboard in a browser.
#
#   ./serve.sh                                       the whole trial, browser opens
#   ./serve.sh --seconds 60 --serve-seconds 900      1 min of replay, 15 min up
#   ./serve.sh --trial datasets/AAD-KULeuven/converted/S1/trial_008.npz
#   ./serve.sh --no-browser --smoke                  start once, print the URL, exit
#
# This is the POSIX twin of scripts/auditory_ui/serve.ps1, which is Windows
# PowerShell and is the reason this file exists: the end goal is this same demo
# running from an external drive on a Mac, where there was no entry point.
#
# WHAT IT ACCEPTS. The flags below are the whole interface; anything else is refused
# with "unknown option" rather than forwarded, so the demo's own --help is not
# available through this script. The flags are --trial, --model, --seconds,
# --serve-seconds, --margin, --eeg-display-channel, --media-owner,
# --url-timeout-seconds, --rebuild, --no-browser, --keep-alive and --smoke.
#
#   sh scripts/auditory_ui/serve.sh --seconds 60 --eeg-display-channel Cz --media-owner page
#
# --eeg-display-channel is what turns the EEG traces on. The producer publishes no
# eeg_display packet unless an electrode is named, so without this flag the page's
# EEG panel reads "Awaiting EEG display data" for the whole run -- honestly, because
# nothing was published. It is off by default because turning a new per-frame packet
# on for every run would change runs nobody asked to change.
#
# --media-owner states who may claim the transport's one media slot: auto, standby,
# page or demo, exactly as the demo's own flag. It is forwarded rather than defaulted
# here because which one is right depends on whether a human is watching the page
# (documents/where_the_demo_stands.md section 5.10).
#
# An earlier version of this comment claimed everything after the flags was passed
# through, and offered `./serve.sh --seconds 30 --margin 0.05
# --eeg-display-channel Cz` as an example. That example did not work then: the
# argument loop ends in `*) fail "unknown option: $1"`, and `set --` below rebuilds
# the command line from scratch. The two flags it named are now real flags, spelled
# out in that loop rather than forwarded blind, which is the difference between a
# wrapper and a hole in one.
# THE INTERPRETER IS DETECTED, NEVER ASSUMED. A virtual environment keeps its
# interpreter at .venv/bin/python on macOS and Linux; this script tries that
# layout first, then python3.13, python3.12, python3 and python on PATH. The
# Windows layout (.venv/Scripts/python.exe) is deliberately NOT probed: a .venv
# cannot be carried between platforms -- the interpreter and every compiled wheel
# are platform-specific -- so on Windows the answer is serve.ps1, not this file.
# If nothing suitable exists this refuses and says how to build one.
#
# IT NEVER RENDERS OVER output/auditory_ui/demo_stereo.wav. That path is
# scripts/auditory_ui/demo.py's own DEFAULT_MEDIA_OUT, and it belongs to whichever
# run claimed it first: the page reads `duration` from the file and the browser
# streams it, so a second run rendering a different trial over that name silently
# re-tunes the session already being served, and two concurrent runs write the
# same bytes through the same name. Every run here renders to its own
# output/auditory_ui/serve_<stamp>.wav instead, and gives --render-out,
# --media-record and --gate-out the same per-run prefix, so a concurrently running
# demo cannot interleave the record a run is cited from. The port is the demo's own
# ephemeral choice -- this script binds nothing and never guesses a port; it reads
# the URL back out of the demo's output.
#
# The stay-open window is the demo's own --serve-seconds, forwarded as
# --serve-seconds and 12 h by default, so the default path starts the demo ONCE
# and waits for it: no restart, and no second URL. --keep-alive restores the older
# behaviour of restarting the demo and printing a new URL, which is opt-in because
# a restart replaces a working page with a new URL and replays from t=0.
#
# Interrupting the default wait with Ctrl-C stops the demo too: without job
# control the child shares this shell's process group, and the demo already
# handles SIGINT/SIGTERM. Under --keep-alive, Ctrl-C stops the loop.
#
# AppleDouble sidecars: on an exFAT volume macOS writes `._name` beside `name`,
# and .DS_Store in every directory it browses. Nothing here globs or counts files,
# so none of them can be mistaken for an input.
#
# Refusals are loud and non-zero, never silent:
#   2  the Python environment is missing or too old, or is not this package
#   1  the demo did not come up, or exited non-zero
#
# Python 3.12 or newer is required (pyproject.toml), on every platform.

set -eu

# 1. Where this repository is, whatever directory the operator is standing in.
#    `CDPATH=` keeps a CDPATH entry from changing what `cd` prints.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO"

fail() {
    printf '\nREFUSED: %s\n' "$1" >&2
    exit 2
}

usage() {
    # Lines 2.. of this file, up to the first line that is not a comment, with the
    # leading `# ` stripped -- the same help the header already is.
    awk 'NR > 1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"
    printf '\nOptions:\n'
    printf '  --trial PATH               converted trial .npz to replay\n'
    printf '  --model PATH               fitted decoder .npz\n'
    printf '  --seconds N               replay only this many seconds; 0 = the whole trial\n'
    printf '  --serve-seconds N         keep serving this long after the replay; 0 = exit with it\n'
    printf '  --margin F                controller commit margin; 0 = the calibrated default\n'
    printf '  --eeg-display-channel C   publish the eeg_display packet for electrode C (e.g. Cz),\n'
    printf '                            which is what draws the two EEG traces\n'
    printf '  --media-owner MODE        auto|standby|page|demo: who may claim the media slot\n'
    printf '  --rebuild                 npm run build apps/attune-ui/dist first\n'
    printf '  --no-browser              do not ask the OS to open a tab\n'
    printf '  --keep-alive              restart the demo when it exits (opt-in)\n'
    printf '  --smoke                   start once with scratch names and exit\n'
    printf '  --url-timeout-seconds N   how long to wait for the URL on each start\n'
    printf '  -h, --help                this text\n'
    printf '\nSee also: scripts/auditory_ui/serve.ps1 (the Windows twin),\n'
    printf '          scripts/getlive/live_demo.sh (the live eego route).\n'
    exit 0
}

# ------------------------------------------------------------------ options ----
TRIAL='datasets/AAD-KULeuven/converted/S1/trial_004.npz'
MODEL='models/auditory_kuleuven.npz'
REPLAY_SECONDS=0
SERVE_SECONDS=43200
MARGIN=0
EEG_DISPLAY_CHANNEL=''
MEDIA_OWNER=''
REBUILD=0
NO_BROWSER=0
KEEP_ALIVE=0
SMOKE=0
URL_TIMEOUT_SECONDS=90

# Values arrive from the operator's own shell, so they are white-listed rather
# than interpolated into a command line unexamined.
number() {
    case "$2" in
        ''|*[!0-9.]*) fail "$1 must be a non-negative number, got '$2'" ;;
    esac
    case "$2" in
        *.*.*) fail "$1 must be a number, got '$2'" ;;
    esac
}
# POSIX sh has no float comparison and --seconds/--margin are doubles; awk does,
# and macOS ships it. `positive` is the `-gt 0` of the PowerShell twin.
positive() { awk -v v="$1" 'BEGIN { exit !(v + 0 > 0) }'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --trial)                [ $# -ge 2 ] || fail "--trial needs a path"; TRIAL=$2; shift 2 ;;
        --trial=*)              TRIAL=${1#*=}; shift ;;
        --model)                [ $# -ge 2 ] || fail "--model needs a path"; MODEL=$2; shift 2 ;;
        --model=*)              MODEL=${1#*=}; shift ;;
        --seconds)              [ $# -ge 2 ] || fail "--seconds needs a number"; number --seconds "$2"; REPLAY_SECONDS=$2; shift 2 ;;
        --seconds=*)            number --seconds "${1#*=}"; REPLAY_SECONDS=${1#*=}; shift ;;
        --serve-seconds)        [ $# -ge 2 ] || fail "--serve-seconds needs a number"; number --serve-seconds "$2"; SERVE_SECONDS=$2; shift 2 ;;
        --serve-seconds=*)      number --serve-seconds "${1#*=}"; SERVE_SECONDS=${1#*=}; shift ;;
        --margin)               [ $# -ge 2 ] || fail "--margin needs a number"; number --margin "$2"; MARGIN=$2; shift 2 ;;
        --margin=*)             number --margin "${1#*=}"; MARGIN=${1#*=}; shift ;;
        --url-timeout-seconds)  [ $# -ge 2 ] || fail "--url-timeout-seconds needs a number"; number --url-timeout-seconds "$2"; URL_TIMEOUT_SECONDS=$2; shift 2 ;;
        --url-timeout-seconds=*) number --url-timeout-seconds "${1#*=}"; URL_TIMEOUT_SECONDS=${1#*=}; shift ;;
        --eeg-display-channel)  [ $# -ge 2 ] || fail "--eeg-display-channel needs an electrode name"; EEG_DISPLAY_CHANNEL=$2; shift 2 ;;
        --eeg-display-channel=*) EEG_DISPLAY_CHANNEL=${1#*=}; shift ;;
        --media-owner)          [ $# -ge 2 ] || fail "--media-owner needs a mode"; MEDIA_OWNER=$2; shift 2 ;;
        --media-owner=*)        MEDIA_OWNER=${1#*=}; shift ;;
        --rebuild)              REBUILD=1; shift ;;
        --no-browser)           NO_BROWSER=1; shift ;;
        --keep-alive)           KEEP_ALIVE=1; shift ;;
        --smoke)                SMOKE=1; shift ;;
        -h|--help|help)         usage ;;
        *)                      fail "unknown option: $1 (try --help)" ;;
    esac
done

# --media-owner reaches the demo verbatim, so it is checked here rather than left
# to fail three layers down. The four modes are the demo's own choices.
case "$MEDIA_OWNER" in
    ''|auto|standby|page|demo) ;;
    *) fail "--media-owner must be auto, standby, page or demo, got '$MEDIA_OWNER'" ;;
esac

# 2. The Python this repository was built with, detected rather than assumed.
#    An explicit ATTUNE_PYTHON wins, so an operator with a different environment
#    can point at it without editing this file.
PYTHON=""
for candidate in \
    "${ATTUNE_PYTHON:-}" \
    "$REPO/.venv/bin/python" \
    "$REPO/.venv/bin/python3" \
    "python3.13" \
    "python3.12" \
    "python3" \
    "python"
do
    [ -n "$candidate" ] || continue
    case "$candidate" in
        */*)
            [ -x "$candidate" ] || continue
            ;;
        *)
            command -v "$candidate" >/dev/null 2>&1 || continue
            ;;
    esac
    PYTHON="$candidate"
    break
done
[ -n "$PYTHON" ] || fail "no Python interpreter found. Tried, in order:
  \$ATTUNE_PYTHON, .venv/bin/python, .venv/bin/python3,
  python3.13, python3.12, python3, python

A .venv cannot be copied between macOS and Windows: the interpreter and every
compiled wheel are platform-specific. Build the environment on THIS machine,
with Python 3.12 or newer:

  macOS/Linux   python3.12 -m venv .venv && .venv/bin/python -m pip install -e .
  Windows       py -3.12 -m venv .venv ; .venv\\Scripts\\python.exe -m pip install -e .

macOS ships an older system Python, so install 3.12+ first if 'python3.12' is
missing:  brew install python@3.12   |   https://www.python.org/downloads/   |
  uv python install 3.12

(or set ATTUNE_PYTHON to an interpreter you already have)"

# 3. New enough, and able to import this package. An explicit version check says
#    which wall was hit instead of letting an import error three levels down
#    explain it.
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' || fail \
    "$PYTHON is $("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'); this repository requires 3.12 or newer. Install one with 'brew install python@3.12', python.org, or 'uv python install 3.12', then recreate the environment:
    python3.12 -m venv .venv && .venv/bin/python -m pip install -e ."
"$PYTHON" -c 'import nova2026' >/dev/null 2>&1 || fail \
    "$PYTHON cannot import nova2026. Install the package into that environment:
    \"$PYTHON\" -m pip install -e ."

# 4. Hand the environment to the child the way THIS shell does it, and put the
#    repository on the import path so `-m scripts....` resolves from any working
#    directory. `:` separates PYTHONPATH entries here, `;` on Windows.
ATTUNE_PYTHON="$PYTHON"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export ATTUNE_PYTHON PYTHONPATH

# 5. Inputs, resolved against the repository rather than the caller's cwd.
case "$TRIAL" in /*) TRIAL_PATH=$TRIAL ;; *) TRIAL_PATH="$REPO/$TRIAL" ;; esac
case "$MODEL" in /*) MODEL_PATH=$MODEL ;; *) MODEL_PATH="$REPO/$MODEL" ;; esac
[ -f "$TRIAL_PATH" ] || fail "trial not found: $TRIAL_PATH"
[ -f "$MODEL_PATH" ] || fail "decoder not found: $MODEL_PATH"

OUT_DIR="$REPO/output/auditory_ui"
mkdir -p "$OUT_DIR"

if [ "$REBUILD" -eq 1 ]; then
    command -v npm >/dev/null 2>&1 || fail "npm not found; cannot rebuild apps/attune-ui/dist"
    printf 'rebuilding the frontend: npm run build (in %s)\n' "$REPO/apps/attune-ui"
    ( cd "$REPO/apps/attune-ui" && npm run build ) ||
        fail "the frontend build failed (npm run build)"
    [ -f "$REPO/apps/attune-ui/dist/index.html" ] ||
        fail "the build produced no apps/attune-ui/dist/index.html"
fi

# 6. Start the repository's own entry point -- scripts.auditory_ui.demo -- with an
#    explicit --media-out, in the background, logging where its URL can be read.
#    Nothing here reimplements the demo: this is a flag-remembering wrapper.
start_demo() {
    RUN_TAG=$1
    STDOUT_LOG="$OUT_DIR/$RUN_TAG.stdout.log"
    STDERR_LOG="$OUT_DIR/$RUN_TAG.stderr.log"
    set -- -B -m scripts.auditory_ui.demo \
        --trial "$TRIAL" \
        --model "$MODEL" \
        --out "output/auditory_ui/$RUN_TAG.json" \
        --stream-out "output/auditory_ui/${RUN_TAG}_packets.jsonl" \
        --media-out "output/auditory_ui/$RUN_TAG.wav" \
        --render-out "output/auditory_ui/${RUN_TAG}_dashboard.html" \
        --media-record "output/auditory_ui/${RUN_TAG}_media.json" \
        --gate-out "output/auditory_ui/${RUN_TAG}_gate.json"
    if positive "$REPLAY_SECONDS"; then set -- "$@" --seconds "$REPLAY_SECONDS"; fi
    if positive "$MARGIN"; then set -- "$@" --margin "$MARGIN"; fi
    if positive "$SERVE_SECONDS" && [ "$SMOKE" -eq 0 ]; then set -- "$@" --serve-seconds "$SERVE_SECONDS"; fi
    # Named explicitly rather than forwarded blind: these two are the whole reason
    # the flags exist above, and an empty one must stay absent from the command.
    if [ -n "$EEG_DISPLAY_CHANNEL" ]; then set -- "$@" --eeg-display-channel "$EEG_DISPLAY_CHANNEL"; fi
    if [ -n "$MEDIA_OWNER" ]; then set -- "$@" --media-owner "$MEDIA_OWNER"; fi
    if [ "$NO_BROWSER" -eq 0 ]; then set -- "$@" --open-browser; fi

    printf '\ncommand  : %s %s\n' "$PYTHON" "$*"
    printf 'stdout   : %s\n' "$STDOUT_LOG"
    printf 'media-out: output/auditory_ui/%s.wav  (never output/auditory_ui/demo_stereo.wav)\n' "$RUN_TAG"

    "$PYTHON" "$@" >"$STDOUT_LOG" 2>"$STDERR_LOG" &
    DEMO_PID=$!
}

# The demo prints "serving the demo at http://127.0.0.1:<port>/" before the replay
# starts, and "open: http://127.0.0.1:<port>/" in its final report. The URL is read
# back rather than assumed because the demo binds an ephemeral port.
wait_for_url() {
    DEMO_URL=""
    waited=0
    while [ "$waited" -lt "$URL_TIMEOUT_SECONDS" ]; do
        if [ -f "$STDOUT_LOG" ]; then
            DEMO_URL=$(grep -o 'http://127\.0\.0\.1:[0-9]\{1,\}/' "$STDOUT_LOG" 2>/dev/null | head -n 1)
        fi
        [ -n "$DEMO_URL" ] && return 0
        kill -0 "$DEMO_PID" 2>/dev/null || return 1
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

show_failure() {
    printf 'the demo did not print a URL within %ss.\n' "$URL_TIMEOUT_SECONDS"
    kill -0 "$DEMO_PID" 2>/dev/null || printf 'it has exited.\n'
    printf '%s\n' '--- stderr tail ---'
    tail -n 25 "$STDERR_LOG" 2>/dev/null || true
    printf '%s\n' '--- stdout tail ---'
    tail -n 25 "$STDOUT_LOG" 2>/dev/null || true
}

stamp() { date +%Y%m%d_%H%M%S; }

if [ "$SMOKE" -eq 1 ]; then RUN_TAG='serve_smoke'; else RUN_TAG="serve_$(stamp)"; fi
start_demo "$RUN_TAG"

if ! wait_for_url; then show_failure; exit 1; fi

printf '\n================================================================\n'
printf '  demo server is up:  %s\n' "$DEMO_URL"
printf '================================================================\n'
printf 'pid       : %s\n' "$DEMO_PID"
if positive "$REPLAY_SECONDS"; then printf 'replay    : %ss\n' "$REPLAY_SECONDS"
else printf 'replay    : the whole trial\n'; fi
if positive "$SERVE_SECONDS"; then printf 'stay-open : %ss after the replay\n' "$SERVE_SECONDS"
else printf 'stay-open : none: the demo exits with the replay\n'; fi
printf 'stop      : kill %s\n' "$DEMO_PID"
printf 'transcript: tail -f %s\n\n' "$STDOUT_LOG"

if [ "$SMOKE" -eq 1 ]; then
    printf 'Started once (--smoke): the server is this shell'"'"'s child and keeps running.\n'
    exit 0
fi

# ------------------------------------------------- wait for the demo to end ----
# The default path. The demo itself serves --serve-seconds on the port it printed,
# so there is nothing to restart; this waits for that window (or for the demo to
# fail) and reports how it ended.
if [ "$KEEP_ALIVE" -eq 0 ]; then
    printf 'The demo serves this URL on its own for its stay-open window; waiting for it\n'
    printf 'to exit. No restart is needed and no second URL will be printed.\n\n'
    set +e
    wait "$DEMO_PID"
    DEMO_CODE=$?
    set -e
    printf '[%s] the demo exited (code %s).\n' "$(date +%H:%M:%S)" "$DEMO_CODE"
    exit "$DEMO_CODE"
fi

# ------------------------------------------------- --keep-alive: start again ----
# Opt-in, and only wanted with a deliberately short stay-open window. A restart
# prints a NEW URL and replays from t=0, while the URL printed above stops being
# served: anyone watching that tab gets a dead page.
printf 'Keeping a page up by restarting the demo (--keep-alive): each restart prints a\n'
printf 'new URL and replays from the beginning, and the previous URL stops being\n'
printf 'served. Ctrl-C stops this loop.\n\n'

while :; do
    set +e
    wait "$DEMO_PID"
    DEMO_CODE=$?
    set -e
    printf '\n[%s] the demo exited (code %s); the page is no longer served.\n' "$(date +%H:%M:%S)" "$DEMO_CODE"
    printf '[%s] transcript: %s\n' "$(date +%H:%M:%S)" "$STDOUT_LOG"

    sleep 2
    start_demo "serve_$(stamp)"
    if ! wait_for_url; then show_failure; exit 1; fi
    printf '[%s] restarted: %s\n' "$(date +%H:%M:%S)" "$DEMO_URL"
done
