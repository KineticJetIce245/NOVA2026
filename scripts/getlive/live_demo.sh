#!/usr/bin/env sh
# The two-command live demo, for macOS and Linux (and Git Bash on Windows).
#
#   ./live_demo.sh source      command 1: is the amplifier publishing what we need?
#   ./live_demo.sh run         command 2: the session in front of the browser
#   ./live_demo.sh calibrate   the audio-to-EEG loopback offset
#   ./live_demo.sh check       list every LSL outlet on this network
#
# Anything after the step is passed straight through, so the manual command in
# documents/live_demo_two_commands.md works here too:
#
#   ./live_demo.sh run --sfreq 500 --source-units uV --open-browser
#   ./live_demo.sh calibrate --live --method cable
#   ./live_demo.sh source --mode bridge
#
# THE INTERPRETER IS DETECTED, NEVER ASSUMED. A virtual environment keeps its
# interpreter at .venv/bin/python on macOS and Linux and at
# .venv/Scripts/python.exe on Windows; this script tries the POSIX layout first,
# then the Windows one (so it also works under Git Bash), then python3 and
# python on PATH. A .venv cannot be carried between platforms - the interpreter
# and every compiled wheel are platform-specific - so if none of them exists,
# this refuses and says how to build one.
#
# Shell setup, in THIS shell's own syntax: `NAME=value command` and `export` are
# POSIX, and `$env:NAME = 'value'` does not exist here. Path separators are `/`,
# never `\`. PYTHONPATH entries are separated by `:` here and by `;` on Windows.
# The PowerShell .ps1 beside this file uses that shell's own forms; the two are
# deliberately not copies of each other.
#
# AppleDouble sidecars: on an exFAT volume macOS writes `._name` beside
# `name`, and `.DS_Store` in every directory it browses. Nothing here globs or
# counts files, so none of them can be mistaken for an input - but any command
# that does (a directory listing, a file census) must ignore `._*` and
# `.DS_Store`, and `documents/live_demo_two_commands.md` says so.
#
# Refusals are loud and non-zero, never silent:
#   2  the Python environment is missing or too old, or the amplifier is not
#      publishing what the run asserts
#   1  the command itself failed
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

# 2. The Python this repository was built with, detected rather than assumed.
#    An explicit ATTUNE_PYTHON wins, so an operator with a different environment
#    can point at it without editing this file.
PYTHON=""
for candidate in \
    "${ATTUNE_PYTHON:-}" \
    "$REPO/.venv/bin/python" \
    "$REPO/.venv/bin/python3" \
    "$REPO/.venv/Scripts/python.exe" \
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
  \$ATTUNE_PYTHON, .venv/bin/python, .venv/bin/python3, .venv/Scripts/python.exe,
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
    "$PYTHON is $("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'); this repository requires 3.12 or newer. Install one with 'brew install python@3.12', python.org, or 'uv python install 3.12', then recreate .venv."
"$PYTHON" -c 'import nova2026' >/dev/null 2>&1 || fail \
    "$PYTHON cannot import nova2026. Install the package into that environment:
    \"$PYTHON\" -m pip install -e ."

# 4. Hand the environment to the child process the way THIS shell does it, and
#    put the repository on the import path so `-m scripts....` resolves from any
#    working directory. `:` separates PYTHONPATH entries here, `;` on Windows.
ATTUNE_PYTHON="$PYTHON"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export ATTUNE_PYTHON PYTHONPATH

run_module() {
    module="$1"
    shift
    printf 'python  : %s\n' "$PYTHON"
    printf 'module  : %s %s\n\n' "$module" "$*"
    set +e
    "$PYTHON" -B -m "$module" "$@"
    code=$?
    set -e
    if [ "$code" -eq 2 ]; then
        printf '\nThat exit code means the amplifier was not found or did not publish
the contract this run asserted. Tick "Application options -> Network Operation ->
Enable LSL EEG streaming" in the eego control software, confirm this host is on
the amplifier network, then run:  ./live_demo.sh check\n' >&2
    fi
    exit "$code"
}

step=${1:-help}
[ $# -gt 0 ] && shift

case "$step" in
    source|1)   run_module scripts.getlive.live_source "$@" ;;
    run|2)      run_module scripts.auditory_ui.live "$@" ;;
    calibrate)  run_module scripts.getlive.calibrate_loopback "$@" ;;
    check)      run_module scripts.getlive.probe --all "$@" ;;
    *)
        sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'
        printf '\nSteps: source | run | calibrate | check   (see documents/live_demo_two_commands.md)\n'
        exit 0
        ;;
esac
