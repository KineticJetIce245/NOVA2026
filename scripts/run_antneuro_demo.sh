#!/usr/bin/env sh
# The ANT Neuro live demo, in one command: amplifier -> decisions -> browser page.
#
#   sh scripts/run_antneuro_demo.sh
#   sh scripts/run_antneuro_demo.sh --sfreq 1000 --source-units V
#
# It runs the documented two commands in order (documents/live_demo_two_commands.md):
# the pre-flight, which only checks that the amplifier is publishing the 20
# electrodes this headset's model needs, and then the session in front of the page.
#
# --sfreq and --source-units are OPERATOR ASSERTIONS about the amplifier. They are
# not advertised over LSL, and the pre-flight checks them against what the outlet
# declares and refuses on a mismatch. The values below are the documented ones for
# the EE-22x rig; override them if the probe disagrees.
#
# The EEG traces are on (--eeg-display-channel Cz). Without that flag no
# eeg_display packet is published and the panel would read "Awaiting EEG display
# data" for the whole session -- honestly, because nothing was published.
#
# No amplifier? Run the rehearsal publisher in ANOTHER terminal and start this
# script with --no-preflight (the publisher runs until Ctrl+C, so it cannot be a
# pre-flight step here):
#     sh scripts/getlive/live_demo.sh source --mode replay      # terminal 1
#     sh scripts/run_antneuro_demo.sh --no-preflight            # terminal 2
#
# Exit 2 means "refused", and the underlying launcher prints which wall was hit.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

fail() {
    printf '\nREFUSED: %s\n' "$1" >&2
    exit 2
}

SFREQ=500
UNITS=uV
PREFLIGHT=1
CHANNEL=${ANT_EEG_DISPLAY_CHANNEL:-Cz}

# Only the leading options below are consumed; the first argument this script does
# not recognise ends the loop and the rest is passed to the demo untouched.
while [ $# -gt 0 ]; do
    case "$1" in
        --sfreq)                 [ $# -ge 2 ] || fail "--sfreq needs a number"; SFREQ=$2; shift 2 ;;
        --sfreq=*)               SFREQ=${1#*=}; shift ;;
        --source-units)          [ $# -ge 2 ] || fail "--source-units needs a unit"; UNITS=$2; shift 2 ;;
        --source-units=*)        UNITS=${1#*=}; shift ;;
        --eeg-display-channel)   [ $# -ge 2 ] || fail "--eeg-display-channel needs a name"; CHANNEL=$2; shift 2 ;;
        --eeg-display-channel=*) CHANNEL=${1#*=}; shift ;;
        --no-preflight)          PREFLIGHT=0; shift ;;
        *) break ;;
    esac
done

# The audio is chosen by a candidate/envelope PAIR. The demos' defaults are the
# operator's own ANT session pair under tmp/, and tmp/ is git-ignored: it does not
# travel with the repository. Checked here rather than letting a missing candidate
# become a refusal several layers down.
CANDIDATE_A=${ANT_CANDIDATE_A:-tmp/antneurodata/audio_files/experiment/left_mono.wav}
CANDIDATE_B=${ANT_CANDIDATE_B:-tmp/antneurodata/audio_files/experiment/right_mono.wav}
ENVELOPE_A=${ANT_ENVELOPE_A:-datasets/audio/left_mono.npz}
ENVELOPE_B=${ANT_ENVELOPE_B:-datasets/audio/right_mono.npz}
for f in "$CANDIDATE_A" "$CANDIDATE_B" "$ENVELOPE_A" "$ENVELOPE_B"; do
    [ -f "$f" ] || fail "the demo's audio input is missing: $f
The candidate WAVs live under tmp/, which git ignores and which therefore does not
travel with the repository. Copy the pair and its two envelopes from the machine
that recorded the ANT session, or point this run at another pair with
ANT_CANDIDATE_A / ANT_CANDIDATE_B / ANT_ENVELOPE_A / ANT_ENVELOPE_B (or the demo's
own --candidate-a/--candidate-b/--envelope-a/--envelope-b)."
done

if [ "$PREFLIGHT" -eq 1 ]; then
    printf 'pre-flight: is the amplifier publishing the 20 electrodes this model needs?\n\n'
    sh "$REPO/scripts/getlive/live_demo.sh" source || exit $?
fi

exec sh "$REPO/scripts/getlive/live_demo.sh" run \
    --sfreq "$SFREQ" --source-units "$UNITS" \
    --eeg-display-channel "$CHANNEL" \
    --open-browser "$@"
