"""The live demo's claims, tested where they can be tested without an amplifier.

Three of the four claims command 2 makes are hardware-free, and this file holds
them to that:

1. **The contract gate refuses, by name.** ``scripts.auditory_ui.live`` compares
   the chain's contract with the model's on ``RidgeDecoder``'s gate keys before
   a port is opened. The test drives the same function command 2 calls, with a
   contract that differs on one gate key, and requires the refusal to name that
   key - and requires a contract that differs only on the two provenance keys to
   pass, because those can never be equalised across rigs (plan sections
   3.11/3.17-4).
2. **Stream-name resolution and 20-electrode selection work against a real LSL
   outlet.** The second half publishes a 24-electrode cap **in reverse montage
   order** over LSL in this process and runs the adapter command 2 uses. A
   positional pick would return the wrong 20 electrodes and the value check
   would see it, so the test is falsifiable rather than decorative (plan section
   6.4 item 3). Where no outlet can be created - a host with no multicast - the
   test reports the publisher's own error instead of passing silently, exactly
   as ``tests/streaming/test_ant_live_route.py`` does.
3. **The policy reaches the record.** The channel-quality policy is a
   declaration, not an inherited default: every flag that changes it must appear
   in the record with the value in force and where that value came from (plan
   section 3.17-1).

There is also the calibration's arithmetic: a click train with a known injected
offset must be measured back, and the measurement must fail loudly rather than
invent a number when there are no clicks to find. That is the part of the
loopback measurement that can be checked without a cable and an amplifier; the
measurement itself is the operator's.
"""

import json
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from scripts.auditory_ui.live import (
    ContractRefused,
    apply_alignment,
    build_parser,
    build_stereo,
    contract_gate,
    media_alignment,
    quality_policy,
)
from scripts.getlive.calibrate_loopback import (
    click_track,
    detect_click_times,
    match_clicks,
    summarise,
)
from scripts.getlive.live_source import (
    model_electrodes,
    parse_electrodes,
    select_columns,
    unit_exponent,
)

MODEL = Path(__file__).resolve().parents[2] / "models" / "auditory_kuleuven_live20.npz"

RATE = 500.0
"""The rig's measured rate (plan section 3.11)."""

CAP = (
    "Fp1", "Fp2", "F9", "F7", "F3", "Fz", "F4", "F8", "F10", "M1",
    "T7", "C3", "C4", "T8", "M2", "Cz", "P7", "P3", "Pz", "P4",
    "P8", "Oz", "O1", "O2",
)
"""The 24 electrodes on the real headset, in datasheet order (section 3.11)."""


def load_model():
    """The 20-channel model, or a skip that names the missing file."""

    from nova2026.auditory.decoder import RidgeDecoder

    if not MODEL.is_file():
        raise unittest.SkipTest(f"model not present on this machine: {MODEL}")
    return RidgeDecoder.load(MODEL)


class ContractGateTests(unittest.TestCase):
    """Command 2 must refuse a source whose processing keys disagree."""

    def setUp(self):
        self.model = load_model()

    def test_a_matching_contract_passes_and_gates_nothing(self):
        record = contract_gate(self.model, dict(self.model.contract))
        self.assertEqual(list(record["gated"]), [])
        self.assertIn("eeg_channels", record["gate_keys"])
        self.assertIn("input_sfreq", record["gate_keys"])

    def test_a_wrong_input_rate_is_refused_and_the_key_is_named(self):
        contract = dict(self.model.contract)
        contract["input_sfreq"] = RATE
        with self.assertRaises(ContractRefused) as caught:
            contract_gate(self.model, contract)
        self.assertIn("input_sfreq", str(caught.exception))
        self.assertIn("input_sfreq", caught.exception.record["gated"])
        # The record it refuses with is written to disk by command 2, so it has
        # to carry both sides of the difference rather than only a sentence.
        self.assertEqual(
            caught.exception.record["gated"]["input_sfreq"]["model"],
            self.model.contract["input_sfreq"],
        )

    def test_a_wrong_unit_convention_is_refused(self):
        contract = dict(self.model.contract)
        contract["units"] = "V"
        with self.assertRaises(ContractRefused) as caught:
            contract_gate(self.model, contract)
        self.assertIn("units", caught.exception.record["gated"])

    def test_a_missing_channel_is_refused(self):
        contract = dict(self.model.contract)
        contract["eeg_channels"] = list(self.model.contract["eeg_channels"])[:-1]
        with self.assertRaises(ContractRefused) as caught:
            contract_gate(self.model, contract)
        self.assertIn("eeg_channels", caught.exception.record["gated"])

    def test_provenance_differences_are_recorded_rather_than_gated(self):
        contract = dict(self.model.contract)
        contract["input_reference"] = "CPz (this rig)"
        contract["upstream_processing"] = "eego control software over LSL"
        record = contract_gate(self.model, contract)
        self.assertEqual(list(record["gated"]), [])
        self.assertEqual(
            sorted(record["provenance"]), ["input_reference", "upstream_processing"]
        )
        self.assertEqual(record["source_contract"]["input_reference"], "CPz (this rig)")


class ElectrodeSelectionTests(unittest.TestCase):
    """The 20 electrodes are chosen by name, never by column position."""

    def test_the_model_supplies_the_default_list(self):
        names = model_electrodes(MODEL)
        self.assertEqual(len(names), 20)
        self.assertEqual(names[0], "Fp1")
        self.assertNotIn("F9", names)  # the cap has it; the model does not
        self.assertEqual(parse_electrodes(None, MODEL), names)
        self.assertEqual(parse_electrodes("Cz,Oz", MODEL), ("Cz", "Oz"))

    def test_a_reversed_montage_still_selects_the_right_columns(self):
        declared = tuple(reversed(CAP))
        wanted = parse_electrodes(None, MODEL)
        columns, missing = select_columns(declared, wanted)
        self.assertEqual(missing, ())
        self.assertEqual(
            tuple(declared[index] for index in columns), wanted
        )
        # The point of the test: on a reversed montage the correct answer is
        # not the first twenty columns, so a positional pick is visible.
        self.assertNotEqual(list(columns), list(range(len(wanted))))

    def test_the_four_extra_electrodes_are_dropped_not_truncated(self):
        wanted = parse_electrodes(None, MODEL)
        columns, missing = select_columns(CAP, wanted)
        self.assertEqual(missing, ())
        self.assertNotIn(CAP.index("F9"), columns)
        self.assertEqual(len(columns), 20)

    def test_a_missing_electrode_is_named(self):
        declared = tuple(name for name in CAP if name not in ("Cz", "Pz"))
        _columns, missing = select_columns(declared, parse_electrodes(None, MODEL))
        self.assertEqual(sorted(missing), ["Cz", "Pz"])

    def test_the_unit_convention(self):
        self.assertEqual(unit_exponent("uV"), -6)
        self.assertEqual(unit_exponent("V"), 0)
        with self.assertRaises(ValueError):
            unit_exponent("furlongs")


class PolicyRecordTests(unittest.TestCase):
    """The quality policy is declared, recorded, and never inherited silently."""

    def test_the_default_policy_is_the_documented_relaxed_one(self):
        policy = quality_policy(build_parser().parse_args([]))
        self.assertFalse(policy["check_channels"])
        self.assertEqual(policy["max_bad_channels"], 0)
        self.assertEqual(policy["exclude_channels"], [])
        self.assertIn("signal_quality", policy["note"])

    def test_strict_policy_restores_the_chain_default(self):
        policy = quality_policy(build_parser().parse_args(["--strict-policy"]))
        self.assertTrue(policy["check_channels"])
        self.assertIn("stops the session", policy["note"])

    def test_every_declared_flag_reaches_the_record(self):
        args = build_parser().parse_args([
            "--exclude-channels", "F8, F3",
            "--max-bad-channels", "2",
            "--amplitude-limit-uv", "2000",
            "--saturation-limit-uv", "60000",
            "--margin", "0.05",
        ])
        policy = quality_policy(args)
        self.assertEqual(policy["exclude_channels"], ["F8", "F3"])
        self.assertEqual(policy["max_bad_channels"], 2)
        self.assertEqual(policy["amplitude_limit_uv"], 2000.0)
        self.assertEqual(policy["saturation_limit_uv"], 60000.0)
        self.assertEqual(policy["margin"], 0.05)
        self.assertIn("declared", policy["amplitude_limit_source"])
        # JSON-serialisable, because the run record is written as JSON.
        json.dumps(policy)

    def test_an_undeclared_limit_is_recorded_as_the_chain_default(self):
        policy = quality_policy(build_parser().parse_args([]))
        self.assertIsNone(policy["amplitude_limit_uv"])
        self.assertIn("chain default", policy["amplitude_limit_source"])
        self.assertIn("chain default", policy["saturation_limit_source"])


class AudioAnchorTests(unittest.TestCase):
    """The anchor the envelopes are indexed on, and its guard."""

    def test_a_media_packet_yields_the_anchor(self):
        packet = {"type": "media", "timestamp": 40.0,
                  "payload": {"media_time_s": 12.5, "playback_state": "playing"}}
        self.assertAlmostEqual(media_alignment(packet), 27.5, places=9)

    def test_a_non_media_packet_yields_nothing(self):
        self.assertIsNone(media_alignment({"type": "attention", "timestamp": 1.0}))
        self.assertIsNone(media_alignment({"type": "media", "payload": {}}))

    def test_an_implausible_anchor_is_refused_rather_than_adopted(self):
        class Source:
            audio_start = 0.0

        source = Source()
        outcome = apply_alignment(source, 1.0e6, limit=60.0)
        self.assertFalse(outcome["applied"])
        self.assertEqual(source.audio_start, 0.0)
        self.assertIn("outside the plausible window", outcome["reason"])

    def test_a_plausible_anchor_is_adopted(self):
        class Source:
            audio_start = 0.0

        source = Source()
        outcome = apply_alignment(source, 0.42, limit=60.0)
        self.assertTrue(outcome["applied"])
        self.assertAlmostEqual(source.audio_start, 0.42, places=9)


class LiveOutletTests(unittest.TestCase):
    """The adapter command 2 uses, against a real outlet on this host."""

    def test_the_twenty_electrodes_arrive_by_name_in_contract_order(self):
        from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

        from scripts.getlive.ant_source import AntStreamSource
        from scripts.getlive.outlets import open_inlet, wait_for_outlet

        wanted = parse_electrodes(None, MODEL)
        declared = tuple(reversed(CAP))  # deliberately NOT the chain's order
        # One constant level per column: after the by-name selection, column i
        # of the delivered block must equal the level of wanted[i].
        levels = np.arange(1.0, len(declared) + 1.0)
        blocks, chunk = 20, 25
        data = np.tile(levels, (blocks * chunk, 1)).astype(np.float32)
        name, source_id = "nova-live-demo-test", "nova-live-demo-1"
        stop = threading.Event()
        sent = {"blocks": 0}
        # The outlet exists before resolution, exactly as the amplifier's does -
        # and as tests/streaming/test_ant_live_route.py arranges it.
        info = StreamInfo(name, "EEG", len(declared), RATE, "float32", source_id)
        info.set_channel_names(list(declared))
        info.set_channel_types(["eeg"] * len(declared))
        info.set_channel_units(["microvolts"] * len(declared))
        try:
            outlet = StreamOutlet(info, chunk_size=chunk)
        except Exception as error:  # pragma: no cover - no multicast on this host
            raise unittest.SkipTest(f"no LSL outlet on this host: {error}")

        def publish() -> None:
            # Publish until the reader stops, not for a fixed count: a fixed
            # count races the reader (LSL replays no history to an inlet that
            # connects later), and a test whose outcome depends on thread timing
            # is not evidence of anything.
            origin = local_clock() + 0.05
            index = 0
            while not stop.is_set():
                start = (index % blocks) * chunk
                outlet.push_chunk(
                    data[start : start + chunk],
                    np.arange(index * chunk, (index + 1) * chunk) / RATE + origin,
                )
                sent["blocks"] += 1
                index += 1
                time.sleep(chunk / RATE)

        thread = threading.Thread(target=publish, daemon=True)
        # Resolution first (the outlet object exists, so it is visible on the
        # network), then the inlet, then the samples: LSL does not replay
        # history to an inlet that connects later, which is exactly the ordering
        # tests/streaming/test_ant_live_route.py uses and the reason the publish
        # thread is started after `open_inlet` rather than before it.
        try:
            wait_for_outlet(name=name, source_id=source_id, stream_type="EEG",
                            timeout=10.0)
        except RuntimeError as error:  # pragma: no cover - no outlet on this host
            raise unittest.SkipTest(f"no LSL outlet on this host: {error}")
        stream = open_inlet({"name": name, "stype": "EEG", "source_id": source_id},
                            bufsize=5.0, connect_timeout=10.0)
        thread.start()
        try:
            source = AntStreamSource(
                stream, channels=wanted, sfreq=RATE, block_samples=chunk,
                no_data_timeout=2.0,
            )
            self.assertEqual(source.channel_names, wanted)
            got = None
            for _end, rows, times in source.chunks(stop):
                got = (np.asarray(rows), np.asarray(times))
                break
        finally:
            stop.set()
            thread.join(timeout=5)
            if stream.connected:
                stream.disconnect()
        self.assertIsNotNone(
            got,
            "no block was delivered; the source ended with "
            f"{source.transport.ended!r} after {source.transport.blocks} block(s)",
        )
        rows, times = got
        self.assertEqual(rows.shape[1], 20)
        # By name: the value on delivered column i is the level the source put
        # on wanted[i], which on this reversed montage is not `i + 1`.
        expected = np.asarray([levels[declared.index(label)] for label in wanted])
        for index, label in enumerate(wanted):
            self.assertEqual(rows[0, index], expected[index],
                             f"column {index} ({label}) carries the wrong electrode")
        self.assertNotEqual(list(rows[0]), list(np.arange(1.0, 21.0)))
        self.assertAlmostEqual(float(times[0]), 0.0, places=9)

    def test_an_absent_stream_name_is_refused_not_waited_for(self):
        from scripts.getlive.outlets import wait_for_outlet

        started = time.monotonic()
        with self.assertRaises(RuntimeError) as caught:
            wait_for_outlet(name="nova-live-demo-not-publishing", timeout=0.6,
                            interval=0.2)
        message = str(caught.exception)
        # Two honest refusals, depending on whether anything else is publishing:
        # "nothing appeared" on a quiet network, "nothing matches that identity"
        # on a busy one (this host has other LSL tests running). Both are
        # refusals; hanging or silently picking another outlet is what the test
        # is here to catch.
        self.assertTrue(
            "No LSL outlet appeared" in message
            or "No outlet matches the requested identity" in message,
            message,
        )
        self.assertLess(time.monotonic() - started, 10.0)


class StereoAssemblyTests(unittest.TestCase):
    """Two candidates of different length are refused, not silently truncated."""

    def test_equal_lengths_stack(self):
        audio, report = build_stereo(
            np.zeros(100), np.ones(100), rate=100, mismatch_seconds=0.001
        )
        self.assertEqual(audio.shape, (100, 2))
        self.assertEqual(report["published_seconds"], 1.0)

    def test_a_length_mismatch_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            build_stereo(np.zeros(100), np.zeros(50), rate=100,
                         mismatch_seconds=0.001)
        self.assertIn("differ", str(caught.exception))


class CalibrationArithmeticTests(unittest.TestCase):
    """The loopback arithmetic, with the amplifier replaced by a known answer."""

    def _synthetic(self, offset, *, clicks=6, rate=500.0, interval=2.0, lead=1.0):
        """EEG carrying the click train shifted by ``offset`` seconds."""

        seconds = lead + clicks * interval + 1.0
        rows = int(seconds * rate)
        eeg = np.random.default_rng(3).normal(0.0, 2.0, (rows, 4))
        scheduled = []
        for index in range(clicks):
            moment = lead + index * interval
            position = int(round((moment + offset) * rate))
            # A click is a step: two samples, opposite sign - what the first
            # difference detector is supposed to find, and nothing like it
            # exists elsewhere in the noise.
            eeg[position, 2] += 300.0
            eeg[position + 1, 2] -= 300.0
            scheduled.append(moment + offset)
        return eeg, tuple(scheduled)

    def test_the_click_track_schedule_is_exact(self):
        _wave, times = click_track(seconds=12.0, rate=48000, interval=2.0, lead=1.0)
        # lead 1 s, then every 2 s while the click still fits in 12 s: 1..11.
        self.assertEqual(list(times), [1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        self.assertAlmostEqual(times[0], 1.0, places=9)

    def test_a_known_offset_comes_back(self):
        # Detection is sample-quantised and, on this deliberately sharp synthetic
        # step, biased by one sample: the first difference crosses at the sample
        # BEFORE the one that moved. Two samples at 500 Hz is the honest
        # tolerance for this shape, and a sign flip or a lost click is far
        # outside it.
        quantum = 2.0 / 500.0
        for injected in (0.0, 0.042, -0.017):
            eeg, scheduled = self._synthetic(injected)
            found = detect_click_times(eeg, 500.0)
            self.assertEqual(len(found["times"]), 6,
                             f"detector lost clicks at offset {injected}")
            matched = match_clicks(scheduled, found["times"], tolerance=0.25)
            record = summarise(matched["pairs"], mismatch=matched["missed"],
                               extra=matched["extra"], interval=2.0)
            self.assertEqual(record["status"], "measured")
            self.assertAlmostEqual(record["offset_seconds"], 0.0, delta=quantum,
                                   msg=f"measured {record['offset_seconds']} for an "
                                       f"injected {injected}")
            self.assertAlmostEqual(record["detected_interval_seconds"], 2.0,
                                   places=3)

    def test_the_sign_convention_is_stated_and_honoured(self):
        eeg, scheduled = self._synthetic(0.0)
        found = detect_click_times(eeg, 500.0)
        shifted = tuple(moment + 0.030 for moment in scheduled)
        matched = match_clicks(shifted, found["times"], tolerance=0.25)
        record = summarise(matched["pairs"], method="cable")
        # The click arrived 30 ms BEFORE the scheduled instant, so the offset is
        # negative: this is the sign a mutation of `seen - moment` would flip,
        # and the flip lands at +0.030, sixteen times the tolerance below.
        self.assertLess(record["offset_seconds"], 0.0)
        self.assertAlmostEqual(record["offset_seconds"], -0.030, delta=2.0 / 500.0)
        self.assertIn("positive means", record["sign_convention"])

    def test_no_clicks_is_not_measurable_rather_than_a_number(self):
        eeg = np.random.default_rng(4).normal(0.0, 2.0, (2500, 4))
        found = detect_click_times(eeg, 500.0)
        _wave, scheduled = click_track(seconds=6.0, rate=48000, interval=2.0, lead=1.0)
        matched = match_clicks(scheduled, found["times"], tolerance=0.25)
        record = summarise(matched["pairs"], mismatch=matched["missed"])
        self.assertEqual(record["status"], "not measurable")
        self.assertIsNone(record["offset_seconds"])
        self.assertIn("no click was matched", record["reason"])

    def test_a_wide_spread_is_reported_as_untrustworthy(self):
        pairs = tuple((float(index), float(index) + (0.001 if index % 2 else 0.2))
                      for index in range(8))
        record = summarise(pairs, tolerance=0.030)
        self.assertIn("spread exceeds", record["status"])
        self.assertGreater(record["uncertainty_seconds"], 0.030)


if __name__ == "__main__":
    unittest.main()
