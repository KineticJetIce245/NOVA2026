#!/usr/bin/env sh
# The synthetic-ANT demo, in one command: the whole live route, no amplifier.
#
#   sh scripts/run_ant_synth_demo.sh
#   sh scripts/run_ant_synth_demo.sh --seconds 300 --signal levels
#   sh scripts/run_ant_synth_demo.sh --no-browser          # for a scripted check
#
# It starts a synthetic EEG outlet (scripts/getlive/ant_synth.py) with the real
# rig's shape -- 24 electrodes in a non-chain order, 500 Hz, microvolts -- runs the
# same pre-flight the hardware gets, and opens the live session in the browser. The
# publisher is stopped when this script exits, however it exits.
#
# WHAT THIS IS NOT. The samples are synthetic and uncorrelated with the speech
# envelopes, so the decoder's scores are not meaningful and the decisions on the
# page are evidence about the plumbing, not about anyone's attention. For decisions
# that mean something with no amplifier, replay a recorded ANT session:
#     .venv/bin/python -B -m scripts.getlive.ant_publish --session <trial.npz> --seconds 130
# and run this demo's command 2 against it (see scripts/run_antneuro_demo.sh
# --no-preflight, which is exactly that arrangement).
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

fail() {
    printf '\nREFUSED: %s\n' "$1" >&2
    exit 2
}

PYTHON=""
for candidate in \
    "${ATTUNE_PYTHON:-}" \
    "$REPO/.venv/bin/python" \
    "python3.13" "python3.12" "python3" "python"
do
    [ -n "$candidate" ] || continue
    case "$candidate" in
        */*) [ -x "$candidate" ] || continue ;;
        *)   command -v "$candidate" >/dev/null 2>&1 || continue ;;
    esac
    PYTHON="$candidate"
    break
done
[ -n "$PYTHON" ] || fail "no Python interpreter found; build .venv on this machine first"

SFREQ=500
UNITS=uV
SIGNAL=osc
CHANNEL=${ANT_EEG_DISPLAY_CHANNEL:-Cz}
OPEN_BROWSER=1
PUBLISH_SECONDS=1800
REPLAY_SECONDS=180

while [ $# -gt 0 ]; do
    case "$1" in
        --signal)                [ $# -ge 2 ] || fail "--signal needs a value"; SIGNAL=$2; shift 2 ;;
        --signal=*)              SIGNAL=${1#*=}; shift ;;
        --seconds)               [ $# -ge 2 ] || fail "--seconds needs a number"; REPLAY_SECONDS=$2; shift 2 ;;
        --seconds=*)             REPLAY_SECONDS=${1#*=}; shift ;;
        --eeg-display-channel)   [ $# -ge 2 ] || fail "--eeg-display-channel needs a name"; CHANNEL=$2; shift 2 ;;
        --eeg-display-channel=*) CHANNEL=${1#*=}; shift ;;
        --no-browser)            OPEN_BROWSER=0; shift ;;
        *) break ;;
    esac
done

case "$SIGNAL" in
    osc|levels) ;;
    *) fail "--signal must be osc or levels, got '$SIGNAL'" ;;
esac

OUT_DIR="$REPO/output/auditory_ui"
mkdir -p "$OUT_DIR"
PUB_LOG="$OUT_DIR/ant_synth_publisher.log"

cleanup() {
    if [ -n "${PUBLISHER_PID:-}" ]; then
        kill "$PUBLISHER_PID" 2>/dev/null || true
        wait "$PUBLISHER_PID" 2>/dev/null || true
        printf '\nsynthetic outlet stopped (pid %s).\n' "$PUBLISHER_PID"
    fi
}
trap cleanup EXIT INT TERM

printf 'starting the synthetic ANT outlet (no amplifier, signal=%s)...\n' "$SIGNAL"
"$PYTHON" -B -m scripts.getlive.ant_synth \
    --signal "$SIGNAL" --seconds "$PUBLISH_SECONDS" > "$PUB_LOG" 2>&1 &
PUBLISHER_PID=$!
printf 'publisher: pid %s, log %s\n' "$PUBLISHER_PID" "$PUB_LOG"

waited=0
while [ "$waited" -lt 30 ]; do
    if grep -q 'publishing' "$PUB_LOG" 2>/dev/null; then break; fi
    kill -0 "$PUBLISHER_PID" 2>/dev/null || fail "the synthetic publisher died; see $PUB_LOG"
    sleep 1
    waited=$((waited + 1))
done
[ "$waited" -lt 30 ] || fail "the synthetic outlet did not come up within 30s; see $PUB_LOG"
printf 'outlet is up after %ss\n\n' "$waited"

printf 'pre-flight: does the live route accept this outlet as the rig?\n\n'
sh "$REPO/scripts/getlive/live_demo.sh" source || exit $?

printf '\nstarting the live session against the synthetic outlet...\n\n'
set -- run \
    --sfreq "$SFREQ" --source-units "$UNITS" \
    --eeg-display-channel "$CHANNEL" \
    --seconds "$REPLAY_SECONDS"
if [ "$OPEN_BROWSER" -eq 1 ]; then set -- "$@" --open-browser; fi
# Deliberately NOT `exec`: this shell has to outlive the session so its EXIT trap
# can stop the synthetic outlet. `exec` replaces the shell, the trap never runs,
# and the publisher is orphaned on the network -- found by running it.
set +e
sh "$REPO/scripts/getlive/live_demo.sh" "$@"
code=$?
set -e
exit "$code"
