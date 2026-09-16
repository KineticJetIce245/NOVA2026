"""The ANT importer's decisions, on fixtures small enough to run anywhere.

The recording itself lives in a git-ignored, read-only directory, so nothing here
reads it, and nothing here skips: every premise is built in the test. What is
transcribed from the recording is only its *structure* - the channel order, the
two sessions' marker lists with their recorded onsets, the audio start offsets -
and the fixture keeps those times while shrinking the sample rate, because the
marker list spans 327 s and the tests are about times, not about 500 Hz.

The channel fixture gives every electrode a distinct value, so a column chosen by
position instead of by name cannot pass: column j of the trial is the value of the
channel the contract names at position j.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.data import load_trial
from nova2026.auditory.envelopes import EnvelopeVerificationError
from scripts.auditory import antneuro
from scripts.auditory import envelopes as envelope_cli

#: Exactly what the .cnt files hold, in file column order (measured; the report
#: in results/antneuro_testset_report.md carries the command).
RECORDED_CHANNELS = (
    "Fp1", "Fp2", "F9", "F7", "F3", "Fz", "F4", "F8", "F10", "M1", "T7",
    "C3", "C4", "T8", "M2", "Cz", "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",
)

#: Session 1's markers as recorded: one Start, twelve left cues, twelve right
#: cues, one impedance event. The cues alternate strictly.
SESSION_1_MARKERS = (
    (5.006, "1004", "Start", 0.0),
    (17.746, "1001", "Left side", 0.0),
    (25.750, "1006", "Custom Annotation", 0.0),
    (33.896, "1001", "Left side", 0.0),
    (44.742, "1006", "Custom Annotation", 0.0),
    (55.874, "1001", "Left side", 0.0),
    (66.832, "1006", "Custom Annotation", 0.0),
    (77.678, "1001", "Left side", 0.0),
    (89.860, "1006", "Custom Annotation", 0.0),
    (101.488, "1001", "Left side", 0.0),
    (114.888, "1006", "Custom Annotation", 0.0),
    (126.056, "1001", "Left side", 0.0),
    (137.506, "1006", "Custom Annotation", 0.0),
    (152.576, "1001", "Left side", 0.0),
    (166.666, "1006", "Custom Annotation", 0.0),
    (179.590, "1001", "Left side", 0.0),
    (193.660, "1006", "Custom Annotation", 0.0),
    (207.478, "1001", "Left side", 0.0),
    (221.824, "1006", "Custom Annotation", 0.0),
    (236.038, "1001", "Left side", 0.0),
    (251.386, "1006", "Custom Annotation", 0.0),
    (264.854, "1001", "Left side", 0.0),
    (276.796, "1006", "Custom Annotation", 0.0),
    (292.124, "1001", "Left side", 0.0),
    (306.304, "1006", "Custom Annotation", 0.0),
    (310.178, "impedance", "impedance", 0.0),
)

#: Session 2's markers as recorded, including the three events the operator
#: called spurious: the 2.000 s Saying-YES at 148.354 s, the right cue 0.048 s
#: later, and the second Start at 326.692 s.
SESSION_2_MARKERS = (
    (12.838, "1004", "Start", 0.0),
    (19.790, "1006", "Custom Annotation", 0.0),
    (32.248, "1001", "Left side", 0.0),
    (45.106, "1006", "Custom Annotation", 0.0),
    (56.028, "1001", "Left side", 0.0),
    (68.356, "1006", "Custom Annotation", 0.0),
    (85.048, "1001", "Left side", 0.0),
    (97.202, "1006", "Custom Annotation", 0.0),
    (111.468, "1001", "Left side", 0.0),
    (123.732, "1006", "Custom Annotation", 0.0),
    (135.554, "1001", "Left side", 0.0),
    (148.354, "1007", "Saying-YES", 2.0),
    (148.402, "1006", "Custom Annotation", 0.0),
    (161.224, "1001", "Left side", 0.0),
    (174.904, "1006", "Custom Annotation", 0.0),
    (187.694, "1001", "Left side", 0.0),
    (203.218, "1006", "Custom Annotation", 0.0),
    (218.338, "1001", "Left side", 0.0),
    (233.948, "1006", "Custom Annotation", 0.0),
    (245.046, "1001", "Left side", 0.0),
    (258.082, "1006", "Custom Annotation", 0.0),
    (270.286, "1001", "Left side", 0.0),
    (282.874, "1006", "Custom Annotation", 0.0),
    (293.962, "1001", "Left side", 0.0),
    (306.388, "1006", "Custom Annotation", 0.0),
    (317.564, "1001", "Left side", 0.0),
    (326.692, "1004", "Start", 0.0),
    (328.626, "impedance", "impedance", 0.0),
)


def annotations(markers):
    """Turn ``(onset, code, name, duration)`` rows into annotation triples."""

    return tuple(
        (float(onset), float(duration), f"{code}/{name}")
        for onset, code, name, duration in markers
    )


def source_for(
    markers,
    session,
    *,
    seconds,
    rate=10.0,
    channels=RECORDED_CHANNELS,
    unit="V",
    data=None,
    path=None,
):
    """A :class:`SessionSource` shaped like the recording, with identifiable columns."""

    samples = int(round(seconds * rate))
    if data is None:
        # Channel k carries (k + 1) uV expressed in volts, so selecting by name is
        # checkable by value and selecting by position cannot pass unnoticed.
        volts = np.arange(1, len(channels) + 1, dtype=float) * 1e-6
        data = np.tile(volts, (samples, 1))
    return antneuro.SessionSource(
        path=Path(path or f"Lacroix_Flo2_2026-09-12_{session}.cnt"),
        session=session,
        sample_rate=rate,
        channel_names=tuple(channels),
        data=np.asarray(data, dtype=float),
        unit=unit,
        annotations=annotations(markers),
    )


class StubEnvelope:
    """The two fields :func:`antneuro.place_envelope` reads, and nothing else."""

    def __init__(self, timestamps, envelope):
        self.timestamps = np.asarray(timestamps, dtype=float)
        self.envelope = np.asarray(envelope, dtype=float).reshape(-1, 1)


def ramp_envelope(seconds=900.0, rate=64.0):
    """An envelope proportional to its own timestamp, so placement is checkable.

    Scaled by 1/1000 to stay inside the trial format's [-1, 1] audio range: the
    value at stimulus position ``p`` is ``p / 1000``, which is an identity the
    placement test can assert instead of a tolerance chosen to make it pass.
    """

    timestamps = np.arange(int(seconds * rate)) / rate
    return StubEnvelope(timestamps, timestamps / 1000.0)


def ramp_envelopes(seconds=900.0):
    """Two distinguishable candidate envelopes for tests that are not about audio."""

    return (ramp_envelope(seconds), ramp_envelope(seconds))


class ChannelContractTests(unittest.TestCase):
    """Twenty electrodes, chosen by name, in the plan's order."""

    def test_the_contract_is_the_twenty_shared_electrodes_in_plan_order(self):
        self.assertEqual(
            antneuro.SHARED_CHANNEL_NAMES,
            (
                "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4",
                "T8", "Cz", "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",
            ),
        )
        self.assertEqual(len(antneuro.SHARED_CHANNEL_NAMES), 20)

    def test_channels_are_selected_by_name_and_extras_are_dropped_by_name(self):
        shuffled = ["Bogus"] + list(reversed(RECORDED_CHANNELS))
        source = source_for(SESSION_1_MARKERS, "19-34-06", seconds=310.19, channels=shuffled)
        trial, summary = antneuro.build_session(source, ramp_envelopes())
        self.assertEqual(trial.channel_names, antneuro.SHARED_CHANNEL_NAMES)
        # Column j must hold the value of the channel the contract names at j.
        expected = np.array([shuffled.index(name) + 1 for name in trial.channel_names])
        np.testing.assert_allclose(trial.eeg[0], expected.astype(float))
        self.assertEqual(
            summary["dropped_channels"], ["Bogus", "M2", "M1", "F10", "F9"]
        )
        self.assertEqual(summary["channels_recorded"], 25)

    def test_a_missing_contract_channel_is_refused_by_name(self):
        without_pz = [name for name in RECORDED_CHANNELS if name != "Pz"]
        source = source_for(SESSION_1_MARKERS, "19-34-06", seconds=20.0, channels=without_pz)
        with self.assertRaises(ValueError) as caught:
            antneuro.build_session(source, ramp_envelopes())
        self.assertIn("Pz", str(caught.exception))

    def test_a_channel_by_samples_array_is_refused_rather_than_misread(self):
        # MNE's own orientation. Selecting by name on it would pick samples, so the
        # shape must be refused before any column is chosen.
        channels = list(reversed(RECORDED_CHANNELS))
        source = source_for(
            SESSION_1_MARKERS,
            "19-34-06",
            seconds=20.0,
            channels=channels,
            data=np.zeros((len(channels), 200)),
        )
        with self.assertRaises(ValueError) as caught:
            antneuro.build_session(source, ramp_envelopes())
        self.assertIn("samples by channels", str(caught.exception))


class UnitTests(unittest.TestCase):
    """Volts in the file, microvolts in the trial, and the metadata says both."""

    def test_volts_become_microvolts_and_the_trial_records_both_units(self):
        source = source_for(SESSION_1_MARKERS, "19-34-06", seconds=20.0)
        trial, summary = antneuro.build_session(source, ramp_envelopes())
        np.testing.assert_allclose(
            trial.eeg[0],
            [RECORDED_CHANNELS.index(name) + 1 for name in antneuro.SHARED_CHANNEL_NAMES],
        )
        self.assertEqual(summary["unit"]["source_unit"], "V")
        self.assertEqual(summary["unit"]["trial_unit"], "uV")
        self.assertEqual(summary["unit"]["scale_to_microvolts"], 1e6)
        self.assertEqual(summary["unit"]["chain_source_unit_exponent"], -6)
        self.assertIn("converted to microvolts by x1e+06", trial.upstream_processing)
        self.assertIn(
            "source_unit_exponent for these values is therefore -6",
            trial.upstream_processing,
        )

    def test_an_unknown_unit_is_refused_rather_than_scaled_by_guess(self):
        with self.assertRaises(ValueError) as caught:
            antneuro.to_microvolts(np.zeros((2, 2)), "furlongs")
        self.assertIn("furlongs", str(caught.exception))


class MarkerTableTests(unittest.TestCase):
    """Every recorded marker is on the record; the exclusions are explicit."""

    def test_every_recorded_marker_appears_in_the_table(self):
        rows, _, _ = antneuro.marker_table(annotations(SESSION_2_MARKERS), "19-41-11")
        self.assertEqual(len(rows), len(SESSION_2_MARKERS))
        self.assertEqual([row["index"] for row in rows], list(range(len(SESSION_2_MARKERS))))
        self.assertEqual(
            [row["onset_seconds"] for row in rows],
            [onset for onset, _, _, _ in SESSION_2_MARKERS],
        )
        dispositions = [row["disposition"] for row in rows]
        self.assertEqual(dispositions.count("cue"), 23)
        self.assertEqual(dispositions.count("ambiguous-cue"), 1)
        self.assertEqual(dispositions.count("excluded"), 2)
        self.assertEqual(dispositions.count("anchor"), 1)
        self.assertEqual(dispositions.count("non-cue"), 1)
        # The disputed right cue is kept, without hiding that it is disputed.
        self.assertEqual(rows[12]["disposition"], "ambiguous-cue")
        self.assertIn("25.670", rows[12]["reason"])

    def test_the_operator_exclusions_are_applied_and_reported(self):
        rows, used, _ = antneuro.marker_table(annotations(SESSION_2_MARKERS), "19-41-11")
        excluded = [row for row in rows if row["disposition"] == "excluded"]
        self.assertEqual(
            [(row["code"], row["onset_seconds"]) for row in excluded],
            [("1007", 148.354), ("1004", 326.692)],
        )
        self.assertEqual(used, {148.354, 326.692})
        cues = antneuro.cues_from_rows(rows)
        self.assertEqual(len(cues), 24)
        self.assertEqual(sum(1 for _, candidate in cues if candidate == 0), 12)
        self.assertEqual(sum(1 for _, candidate in cues if candidate == 1), 12)
        # Session 2's first cue is right, not left: a builder assuming otherwise
        # would mislabel the whole session.
        self.assertEqual(cues[0], (19.790, 1))

    def test_an_exclusion_that_matches_no_marker_is_an_error(self):
        without_yes = tuple(row for row in SESSION_2_MARKERS if row[1] != "1007")
        with self.assertRaises(ValueError) as caught:
            antneuro.marker_table(annotations(without_yes), "19-41-11")
        self.assertIn("1007@148.354", str(caught.exception))

    def test_a_broken_alternation_is_recorded_and_not_hidden(self):
        repeated = (
            (1.0, "1004", "Start", 0.0),
            (5.0, "1001", "Left side", 0.0),
            (15.0, "1001", "Left side", 0.0),
            (25.0, "1006", "Custom Annotation", 0.0),
        )
        rows, _, _ = antneuro.marker_table(annotations(repeated), "19-34-06")
        report = antneuro.alternation_report(antneuro.cues_from_rows(rows))
        self.assertFalse(report["ok"])
        self.assertEqual(report["breaks"], [{"after_seconds": 5.0, "at_seconds": 15.0, "candidate": 0}])


class LabelRuleTests(unittest.TestCase):
    """The operator's buffer rule, including its endpoints and its collisions."""

    def test_the_buffer_is_symmetric_and_includes_its_endpoints(self):
        times = np.arange(0, 20.5, 0.5)
        cues = [(10.0, 0), (16.0, 1)]
        labels = antneuro.labels_from_cues(times, cues, 0.5)
        at = dict(zip(times.tolist(), labels.tolist()))
        self.assertEqual(at[9.0], -1)
        self.assertEqual(at[9.5], -1)   # the endpoint before the switch
        self.assertEqual(at[10.0], -1)  # the switch itself
        self.assertEqual(at[10.5], -1)  # the endpoint after the switch
        self.assertEqual(at[11.0], 0)   # the first sample that is the new candidate
        self.assertEqual(at[15.5], -1)
        self.assertEqual(at[16.5], -1)
        self.assertEqual(at[17.0], 1)
        self.assertEqual(at[0.0], -1)   # nothing was asked before the first cue

    def test_cues_closer_than_twice_the_buffer_leave_no_label_for_the_first(self):
        times = np.arange(0, 20.0, 0.5)
        cues = [(10.0, 0), (10.6, 1)]
        labels = antneuro.labels_from_cues(times, cues, 0.5)
        segments = antneuro.assigned_segments(times, cues, labels, 2.0)
        self.assertEqual(segments[0]["assigned_seconds"], 0.0)
        self.assertGreater(segments[1]["assigned_seconds"], 0.0)
        self.assertNotIn(0, labels.tolist())

    def test_a_buffer_outside_the_operators_range_is_refused(self):
        times = np.arange(0, 20.0, 0.5)
        for buffer_seconds in (0.25, 1.5):
            with self.assertRaises(ValueError):
                antneuro.labels_from_cues(times, [(10.0, 0)], buffer_seconds)

    def test_the_ambiguous_cue_labels_its_own_side_through_a_whole_session(self):
        source = source_for(SESSION_2_MARKERS, "19-41-11", seconds=328.63, rate=10.0)
        trial, summary = antneuro.build_session(source, ramp_envelopes())
        self.assertEqual(summary["markers"]["candidate_b"], 12)
        self.assertEqual(summary["alternation"]["ok"], True)
        # 148.402 is the disputed right cue: a sample just after its buffer is B,
        # and the 2.000 s Saying-YES event never becomes a cue of its own.
        index = int(round((148.402 + 0.5) * 10)) + 1
        self.assertEqual(trial.labels[index], 1)
        self.assertEqual(trial.labels[int(round(148.354 * 10))], -1)
        # The last cue stands to the end of the recording, and the seconds add up.
        self.assertEqual(trial.labels[-1], 0)
        self.assertAlmostEqual(
            summary["labels"]["a_seconds"] + summary["labels"]["b_seconds"]
            + summary["labels"]["unknown_seconds"],
            summary["duration_seconds"],
            delta=0.002,
        )
        self.assertEqual(summary["trial_count"], 1)

    def test_the_conservative_buffer_is_reported_beside_the_chosen_one(self):
        source = source_for(SESSION_2_MARKERS, "19-41-11", seconds=328.63, rate=10.0)
        _, summary = antneuro.build_session(source, ramp_envelopes())
        _, conservative = antneuro.build_session(
            source, ramp_envelopes(), buffer_seconds=1.0
        )
        self.assertEqual(summary["labels"]["buffer_seconds"], 0.5)
        self.assertEqual(conservative["labels"]["buffer_seconds"], 1.0)
        self.assertLess(
            conservative["labels"]["surviving_seconds"],
            summary["labels"]["surviving_seconds"],
        )
        self.assertAlmostEqual(
            conservative["labels"]["surviving_seconds"],
            summary["labels"]["surviving_seconds_at_buffer_1_0"],
            places=3,
        )


class AudioPlacementTests(unittest.TestCase):
    """The Start anchor and the operator's offset, on the trial clock."""

    def test_audio_is_placed_through_the_start_anchor_and_the_session_offset(self):
        source = source_for(SESSION_2_MARKERS, "19-41-11", seconds=20.0, rate=500.0)
        trial, summary = antneuro.build_session(source, ramp_envelopes())
        audio_rate = float(AuditoryConfig().sample_rate)
        positions = 267.0 + (np.arange(len(trial.audio)) / audio_rate - 12.838)
        inside = positions >= 267.0  # the played range starts at the session offset
        # The ramp envelope's value is its timestamp / 1000, so this is an identity
        # check on the placement map rather than a tolerance chosen to pass.
        np.testing.assert_allclose(trial.audio[inside, 0], positions[inside] / 1000.0)
        np.testing.assert_allclose(trial.audio[inside, 1], positions[inside] / 1000.0)
        self.assertEqual(summary["audio"]["start_offset_seconds"], 267.0)
        self.assertEqual(summary["audio"]["start_anchor_seconds"], 12.838)

    def test_positions_before_playback_are_zero_and_counted_as_uncovered(self):
        source = source_for(SESSION_2_MARKERS, "19-41-11", seconds=20.0, rate=500.0)
        trial, summary = antneuro.build_session(source, ramp_envelopes())
        audio_rate = float(AuditoryConfig().sample_rate)
        uncovered = int(np.ceil(12.838 * audio_rate))
        np.testing.assert_array_equal(trial.audio[:uncovered], np.zeros((uncovered, 2)))
        self.assertEqual(summary["audio"]["covered_samples"], len(trial.audio) - uncovered)
        self.assertLess(
            abs(summary["audio"]["first_stimulus_position_seconds"] - 267.0), 1 / audio_rate
        )
        # Nothing was asked of the participant before the anchor either.
        self.assertTrue(np.all(trial.labels[: int(12.838 * 500)] == -1))

    def test_the_playable_source_range_is_recorded_for_the_renderer(self):
        source = source_for(SESSION_2_MARKERS, "19-41-11", seconds=328.63, rate=10.0)
        _, summary = antneuro.build_session(source, ramp_envelopes())
        start, stop = summary["audio"]["playable_source_slice_samples"]
        self.assertEqual(start, int(round(267.0 * 48000)))
        self.assertEqual(stop, int(round((267.0 + summary["duration_seconds"]) * 48000)))


class RailTests(unittest.TestCase):
    """A railed electrode is measured and reported, never quietly dropped."""

    def test_a_railed_channel_is_measured_reported_and_not_removed(self):
        rate, seconds = 10.0, 20.0
        samples = int(round(seconds * rate))
        volts = np.arange(1, len(RECORDED_CHANNELS) + 1, dtype=float) * 1e-6
        data = np.tile(volts, (samples, 1))
        railed = RECORDED_CHANNELS.index("F8")
        data[:, railed] = 0.0833
        source = source_for(
            SESSION_1_MARKERS, "19-34-06", seconds=seconds, rate=rate, data=data
        )
        trial, summary = antneuro.build_session(source, ramp_envelopes())
        self.assertIn("F8", trial.channel_names)
        self.assertIn(
            {"name": "F8", "railed_fraction": 1.0}, summary["railed_channels"]
        )
        self.assertEqual(
            [row["name"] for row in summary["railed_channels"]], ["F8"]
        )


class RefusalTests(unittest.TestCase):
    """What the importer refuses, and how loudly."""

    def test_a_session_without_an_operator_offset_is_refused(self):
        source = source_for(SESSION_1_MARKERS, "20-00-00", seconds=20.0)
        with self.assertRaises(ValueError) as caught:
            antneuro.build_session(source, ramp_envelopes())
        self.assertIn("20-00-00", str(caught.exception))
        self.assertIn("SESSION_AUDIO_START_SECONDS", str(caught.exception))

    def test_an_unknown_file_name_cannot_name_a_session(self):
        with self.assertRaises(ValueError):
            antneuro.session_key_from_path("recording.cnt")
        self.assertEqual(
            antneuro.session_key_from_path("Lacroix_Flo2_2026-09-12_19-41-11.cnt"),
            "19-41-11",
        )

    def test_a_missing_candidate_envelope_refuses_the_import_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            envelope_dir = root / "audio"
            envelope_dir.mkdir()
            write_envelope(envelope_dir / "left_mono", root, frequency=320.0)
            source = source_for(SESSION_1_MARKERS, "19-34-06", seconds=2.0, rate=500.0)
            with self.assertRaises(EnvelopeVerificationError) as caught:
                antneuro.import_session(source, root / "out", envelope_dir)
            self.assertIn("right_mono.npz", str(caught.exception))
            self.assertFalse((root / "out").exists())


def write_envelope(target_stem, root, *, frequency, rate=8000, seconds=6.0):
    """A real, verifiable envelope fixture: the importer's own loader must accept it."""

    times = np.arange(int(rate * seconds)) / rate
    samples = 0.4 * np.sin(2 * np.pi * frequency * times) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * times))
    wav_path = Path(f"{target_stem}.wav")
    wavfile.write(wav_path, rate, np.round(samples * 32767).astype(np.int16))
    config = AuditoryConfig()
    envelope, timestamps, metadata = envelope_cli.convert_audio(wav_path, config)
    envelope_cli.save_envelope(Path(f"{target_stem}.npz"), envelope, timestamps, metadata)
    return Path(f"{target_stem}.npz")


class InterchangeTests(unittest.TestCase):
    """The written trial is the repository's format, and carries its own provenance."""

    def test_the_written_trial_loads_back_with_the_contract_and_its_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            envelope_dir = root / "audio"
            envelope_dir.mkdir()
            write_envelope(envelope_dir / "left_mono", root, frequency=320.0, seconds=10.0)
            write_envelope(envelope_dir / "right_mono", root, frequency=1500.0, seconds=10.0)
            markers = (
                (0.5, "1004", "Start", 0.0),
                (1.0, "1001", "Left side", 0.0),
                (3.0, "1006", "Custom Annotation", 0.0),
            )
            source = source_for(markers, "19-34-06", seconds=8.0, rate=500.0)
            trial, summary = antneuro.import_session(source, root / "out", envelope_dir)
            restored = load_trial(Path(summary["trial_file"]))
            self.assertEqual(restored.channel_names, antneuro.SHARED_CHANNEL_NAMES)
            self.assertEqual(restored.eeg.shape, (4000, 20))
            self.assertEqual(restored.audio.shape[1], 2)
            self.assertEqual(restored.audio_rate, 64.0)
            self.assertEqual(restored.subject, "ANT")
            self.assertEqual(restored.group, "antneuro|19-34-06")
            np.testing.assert_array_equal(restored.labels, trial.labels)
            np.testing.assert_allclose(restored.eeg, trial.eeg)
            # Cue at 1.0 s, next cue at 3.0 s: A between the two buffers, then B.
            self.assertEqual(restored.labels[750], -1)   # t = 1.5 s, the endpoint
            self.assertEqual(restored.labels[800], 0)    # t = 1.6 s
            self.assertEqual(restored.labels[1500], -1)  # t = 3.0 s, the switch
            self.assertEqual(restored.labels[2000], 1)   # t = 4.0 s
            self.assertEqual(restored.labels[-1], 1)
            self.assertIn("CPz", restored.reference)
            self.assertIn("ANT Neuro recording", restored.upstream_processing)
            self.assertTrue(Path(summary["marker_table_file"]).is_file())
            self.assertIn("1006", Path(summary["marker_table_file"]).read_text(encoding="utf-8"))

    def test_the_summary_reports_shape_duration_and_the_label_distribution(self):
        source = source_for(SESSION_1_MARKERS, "19-34-06", seconds=310.19, rate=10.0)
        trial, summary = antneuro.build_session(source, ramp_envelopes())
        self.assertEqual(summary["shape"], [3102, 20])
        self.assertEqual(summary["sample_rate_hz"], 10.0)
        # n / rate, the convention the recording report uses (310.19 s at 500 Hz).
        self.assertEqual(summary["duration_seconds"], 310.2)
        labels = summary["labels"]
        self.assertAlmostEqual(
            labels["a_seconds"] + labels["b_seconds"] + labels["unknown_seconds"],
            summary["duration_seconds"],
            delta=0.002,
        )
        self.assertEqual(
            labels["surviving_fraction"],
            round((labels["a_seconds"] + labels["b_seconds"]) / 310.2, 4),
        )
        # 17.746 s of header before the first cue plus 24 switches x 1.0 s of buffer
        # out of 310.2 s: about 0.867 survives, and a broken buffer or a dropped
        # switch moves this number.
        self.assertGreater(labels["surviving_fraction"], 0.85)
        self.assertLess(labels["surviving_fraction"], 0.88)
        self.assertEqual(len(labels["segments"]), 24)
        self.assertEqual(summary["trial_count"], 1)

    def test_the_review_document_carries_every_marker_and_the_checks(self):
        source = source_for(SESSION_2_MARKERS, "19-41-11", seconds=328.63, rate=10.0)
        _, summary = antneuro.build_session(source, ramp_envelopes())
        report = antneuro.collect_report(
            [summary], buffer_seconds=0.5, command="python -B -m scripts.auditory.antneuro",
            stamp="19700101-000000",
        )
        text = antneuro.render_markdown(report)
        for row in SESSION_2_MARKERS:
            self.assertIn(f"{row[0]:.3f}", text)
        self.assertIn("148.402", text)
        self.assertIn("ambiguous-cue", text)
        self.assertIn("What to check by hand", text)
        self.assertIn("railed", text.lower())
        self.assertEqual(report["total_trials"], 1)
        self.assertEqual(len(report["exclusions"]), 2)


if __name__ == "__main__":
    unittest.main()
