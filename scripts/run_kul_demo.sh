#!/usr/bin/env sh
# The no-hardware KU Leuven replay demo, in one command.
#
# Replays a real recorded EEG through the real chain at 1x speed and serves the
# page: no amplifier, no participant, no hardware. This is a wrapper, not a second
# implementation -- it calls scripts/auditory_ui/serve.sh and supplies only the two
# flags that launcher does not default.
#
#   sh scripts/run_kul_demo.sh
#   sh scripts/run_kul_demo.sh --seconds 389
#   sh scripts/run_kul_demo.sh --trial datasets/AAD-KULeuven/converted/S1/trial_008.npz
#
# The EEG traces are on (--eeg-display-channel Cz) and the page owns the media slot
# outright (--media-owner page, with the launcher's own --open-browser). That pair
# is what draws the two traces AND plays the audio: without the first the panel
# honestly reads "Awaiting EEG display data", and without the second the demo's
# stand-in can win the one media slot and the page then produces no sound at all
# (documents/where_the_demo_stands.md section 5.10).
#
# Anything you pass is appended after these defaults and reaches serve.sh, whose
# parser takes the LAST occurrence of a repeated flag, so --seconds 389 wins.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

CHANNEL=${KUL_EEG_DISPLAY_CHANNEL:-Cz}

exec sh "$REPO/scripts/auditory_ui/serve.sh" \
    --seconds "${KUL_SECONDS:-60}" \
    --serve-seconds "${KUL_SERVE_SECONDS:-300}" \
    --eeg-display-channel "$CHANNEL" \
    --media-owner page \
    "$@"
