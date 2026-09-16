"""The ``eeg_display`` tap: raw before the band-pass, beside what the decoder eats.

The demo page's EEG panel used to say "Awaiting EEG display data" while every
other panel filled, because ``documents/auditory_ui_protocol.md`` recorded
``eeg_display`` as a type nothing publishes. This module pins what publishing it
costs and what it must not cost.

Five properties, each one a way the feature could be wrong while looking right:

1. **The contract gains no key.** ``AuditoryProcessor.contract`` is compared
   key-by-key against a trained model's, so one new key invalidates
   ``models/auditory_kuleuven.npz`` and ``models/auditory_kuleuven_live20.npz``
   alike. The tap is run policy; the exact key set is pinned with it on and off.
2. **``samples`` is genuinely pre-band-pass.** The raw signal exists only between
   ``Repair`` and ``bandpass`` inside ``feed``; everything after is 1-9 Hz. A
   copied ``window.data`` would pass every shape and type check and be a lie, so
   the test measures a 30 Hz component that only a real pre-band-pass capture
   keeps.
3. **The captured array is copied, not referenced.** The array in flight is
   handed to the band-pass and the resampler, and ``CircularBuffer.push`` returns
   views of storage it reuses. A reference would be silently overwritten before
   the frame that needs it was built.
4. **Decimation is computed, not discovered.** The trace length is
   ``window_length // factor`` for a factor that divides the window exactly, and
   the decimation anti-aliases: a bare ``[::factor]`` slice folds 30 Hz into the
   2 Hz bin of a 128 -> 16 Hz display, which is inside the band the decoder uses.
5. **Off is off.** With no display channel, no field is populated and no packet
   is produced; the encoded size is measured through the real packet builder
   rather than estimated, because exceeding ``MAX_PACKET_BYTES`` makes the
   transport reject the packet outright.

The chain here is a real :class:`AuditoryProcessor` over a synthetic microvolt
trace, so nothing is inherited from ``models/`` or ``datasets/``.
"""

import json
import unittest
from types import SimpleNamespace

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.producer import AttentionProducer
from nova2026.auditory.session import RunPolicy, SessionFrame
from nova2026.auditory.streaming import (
    DISPLAY_TARGET_POINTS,
    AuditoryProcessor,
    RawDisplayRing,
    capture_display,
    decimate_trace,
    decimation_factor,
    shared_decimation,
    stream_config,
    unit_label,
)
from nova2026.transport.protocol import MAX_PACKET_BYTES, make_packet

RATE = 128.0
OUTPUT_RATE = 64.0
CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4",
    "T8", "Cz", "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",
)
SECONDS = 12.0
"""Long enough for two 5 s windows at the model's one-second step."""

PAYLOAD_FIELDS = {
    "sample_rate", "channels", "samples", "filtered_samples",
    "filtered_sample_rate", "original_units", "filtered_units",
    "original_scale_hint_uv", "filtered_scale_hint_uv", "window_seconds",
    "lag_seconds", "channel_source",
}
"""The frozen ``eeg_display`` field set, repeated here so a change has to be made twice."""

CONTRACT_KEYS = (
    "eeg_channels", "output_sfreq", "input_sfreq", "units", "input_reference",
    "upstream_processing", "bandpass", "filter_order", "resample_quality",
    "stage", "window_seconds", "step_seconds",
)
"""The twelve keys ``RidgeDecoder.validate`` compares, recorded from the baseline
before the display tap existed. A thirteenth key would invalidate every trained
model rather than merely changing this chain."""


def settings(**overrides):
    """Chain settings for this fixture: 20 named electrodes at 128 Hz."""

    trial = SimpleNamespace(
        sample_rate=RATE, channel_names=CHANNELS, reference="synthetic fixture",
        upstream_processing="synthetic fixture",
    )
    return stream_config(
        trial, AuditoryConfig(), history=5.0, step=1.0,
        check_channels=bool(overrides.pop("check_channels", False)),
        source_unit_exponent=-6, **overrides,
    )


def trace(seconds=SECONDS, rate=RATE, channels=len(CHANNELS), seed=7, tone_hz=None):
    """A deterministic microvolt trace; ``tone_hz`` adds one out-of-band tone."""

    rng = np.random.default_rng(seed)
    count = int(round(seconds * rate))
    time = np.arange(count) / rate
    rows = [
        (12.0 + index) * np.sin(2 * np.pi * (2.0 + index / 8.0) * time + index)
        + rng.normal(0.0, 2.0, count)
        for index in range(channels)
    ]
    if tone_hz is not None:
        rows[CHANNELS.index("Cz")] += 100.0 * np.sin(2 * np.pi * tone_hz * time)
    return np.stack(rows, axis=1), time


def chain_windows(processor, eeg, time, block=256):
    """Every window a real chain produces from one trace, in order."""

    windows = []
    for start in range(0, len(eeg), block):
        windows.extend(processor.feed((eeg[start : start + block],
                                       time[start : start + block])))
    return windows


def tone_amplitude(values, rate, frequency):
    """Least-squares amplitude of one frequency in one trace."""

    grid = np.arange(len(values)) / rate
    basis = np.stack([np.cos(2 * np.pi * frequency * grid),
                      np.sin(2 * np.pi * frequency * grid)], axis=1)
    coefficients, *_ = np.linalg.lstsq(basis, np.asarray(values, dtype=float), rcond=None)
    return float(np.hypot(*coefficients))


def bare_producer() -> tuple[AttentionProducer, list]:
    """A producer with nothing but packet plumbing, and the kinds it published."""

    producer = AttentionProducer.__new__(AttentionProducer)
    producer.source = "nova-aad"
    producer.simulated = False
    producer.media_reference = None
    producer._last_evidence = 0
    producer._last_reasons = None
    published: list = []
    return producer, published


def frame_with(display: dict) -> SessionFrame:
    """One frame, with or without a display payload."""

    return SessionFrame(
        timestamp=1.0, decision="uncertain", correlation_a=None, correlation_b=None,
        gain_a_db=0.0, gain_b_db=0.0, quality=None, artifact=None, reasons=(),
        window_end=None, evidence_count=0, media_time_s=None, display=display,
    )


class TestContractUnchanged(unittest.TestCase):
    """The display tap is run policy: the contract's key set may not move.

    Mutation that makes this red: adding ``display_channel`` (or any other key)
    to the dict literal in ``AuditoryProcessor.__init__``.
    """

    def test_exact_key_set_with_display_off(self):
        processor = AuditoryProcessor(settings())
        self.assertEqual(tuple(processor.contract), CONTRACT_KEYS)

    def test_exact_key_set_with_display_on(self):
        processor = AuditoryProcessor(settings(display_channel="Cz"))
        self.assertEqual(tuple(processor.contract), CONTRACT_KEYS)
        # And the feature really is on, so the pin above is not vacuous.
        self.assertEqual(processor._display_channel, "Cz")


class TestRawIsPreBandPass(unittest.TestCase):
    """``window.raw`` must predate the filter, and ``window.display`` must say so.

    Mutation that makes this red: replacing ``data[:, index].copy()`` with
    ``rows[:, index]``, or copying ``window.data`` into ``raw`` - the 30 Hz
    component then reads as attenuated in the "raw" trace and the ratio collapses
    from ~1 to ~0.
    """

    @classmethod
    def setUpClass(cls):
        cls.eeg, cls.time = trace(tone_hz=30.0)
        cls.processor = AuditoryProcessor(settings(display_channel="Cz"))
        # Small chunks, as the real source feeds it (ReplaySource's own default is
        # 0.032 s): the packet's own `available_at` is the newest raw sample, and a
        # two-second block would leave the displayed window a second behind it.
        cls.window = chain_windows(cls.processor, cls.eeg, cls.time, block=4)[-1]
        cls.index = CHANNELS.index("Cz")

    def test_raw_keeps_the_out_of_band_tone_and_data_rejects_it(self):
        raw = np.asarray(self.window.raw, dtype=float)
        filtered = np.asarray(self.window.data[:, self.index], dtype=float)
        source = self.eeg[:, self.index]

        # The copy really is the input over one whole window, sample for sample,
        # and it sits at the chain's own clock: its end is the window past the
        # newest raw sample (`available_at`), a resampler delay later than the
        # filtered window's own end -- the distance the packet publishes as
        # `lag_seconds`. The start index is found rather than assumed, so this
        # pins the trace's alignment instead of taking it on faith.
        self.assertEqual(len(raw), round(self.window.display["window_seconds"] * RATE))
        windows = np.lib.stride_tricks.sliding_window_view(source, len(raw))
        matches = np.flatnonzero((windows == raw).all(axis=1))
        self.assertEqual(len(matches), 1, "the raw window does not match the input once")
        start = int(matches[0])
        # The raw window ends at the newest raw sample within the resampler's own
        # group delay (a few samples at 128 Hz), never a whole window away.
        offset = int(round(float(self.window.available_at) * RATE)) - (start + len(raw))
        self.assertLessEqual(abs(offset), 8, f"raw window ends {offset} samples from available_at")

        in_raw = tone_amplitude(raw, RATE, 30.0)
        in_filtered = tone_amplitude(filtered, OUTPUT_RATE, 30.0)
        self.assertGreater(in_raw, 100.0)
        self.assertLess(in_filtered, 5.0)
        self.assertGreater(in_raw / max(in_filtered, 1e-9), 20.0)

    def test_display_carries_both_traces_over_the_same_interval(self):
        """Both traces cover one interval, on one axis, with one point count.

        This is the check that kills "keep the last chunk": a raw trace of four
        samples drawn beside a five-second filtered trace is two different
        intervals presented as a comparison. Mutation that makes it red: capturing
        the chunk (or the last ``chunksize`` samples) instead of cutting the raw
        ring to the window's own interval - the two lengths then differ by the
        window-to-chunk ratio.
        """

        payload = self.window.display
        raw_points = len(payload["samples"])
        filtered_points = len(payload["filtered_samples"])
        self.assertEqual(raw_points, filtered_points)
        self.assertEqual(len(payload["channels"]), 1)
        for row in payload["samples"] + payload["filtered_samples"]:
            self.assertEqual(len(row), len(payload["channels"]))
        self.assertGreater(raw_points, 40)

        # The raw trace is a whole window of raw samples, three orders of
        # magnitude more than one 0.032 s chunk (4 samples at this rate).
        self.assertEqual(len(self.window.raw), round(payload["window_seconds"] * RATE))
        self.assertGreater(len(self.window.raw), 100)

        # Both point counts are the same interval at the same drawn rate.
        drawn = raw_points / payload["window_seconds"]
        self.assertAlmostEqual(drawn, filtered_points / payload["window_seconds"], places=6)
        self.assertAlmostEqual(drawn, RATE / 8.0, places=6)

    def test_display_carries_both_traces_with_their_own_rates(self):
        # Its own chain: the raw trace's alignment to the source is asserted in the
        # test above, and a fixture shared with it would make this test's numbers
        # depend on that one having run.
        #
        # 25 Hz, not 30: the display's box decimation by 8 has its nulls at 16 and
        # 32 Hz, so a 30 Hz tone is almost exactly cancelled by the averaging
        # itself (measured 7.1 of 100 uV) and could not show that the *chain* kept
        # it. 25 Hz sits off that null (measured 21.3 of 100 uV) and folds to 7 Hz.
        tone_hz = 25.0
        eeg, time = trace(tone_hz=tone_hz)
        window = chain_windows(
            AuditoryProcessor(settings(display_channel="Cz")), eeg, time, block=4
        )[-1]
        payload = window.display
        self.assertEqual(payload["sample_rate"], RATE)
        self.assertEqual(payload["filtered_sample_rate"], OUTPUT_RATE)
        self.assertNotEqual(payload["sample_rate"], payload["filtered_sample_rate"])
        raw_trace = np.asarray(payload["samples"], dtype=float).reshape(-1)
        filtered_trace = np.asarray(payload["filtered_samples"], dtype=float).reshape(-1)
        raw_drawn = len(raw_trace) / payload["window_seconds"]
        filtered_drawn = len(filtered_trace) / payload["window_seconds"]
        drawn = RATE / 8.0
        alias = abs(tone_hz - round(tone_hz / drawn) * drawn)
        in_raw = tone_amplitude(raw_trace, raw_drawn, alias)
        in_filtered = tone_amplitude(filtered_trace, filtered_drawn, alias)
        # The displayed raw trace still holds the tone, folded to its alias by the
        # honest box decimation and attenuated only by the box's own sinc.
        self.assertGreater(in_raw, 15.0)
        # ...and the displayed filtered trace does not: the chain's 1-9 Hz
        # band-pass removed it before the window ever reached the display.
        self.assertLess(in_filtered, 2.0)
        self.assertGreater(in_raw / max(in_filtered, 1e-9), 10.0)

    def test_units_and_span_hints_come_from_the_declared_exponent(self):
        payload = self.window.display
        self.assertEqual(payload["original_units"], unit_label(-6))
        self.assertEqual(payload["original_units"], "uV")
        self.assertEqual(payload["filtered_units"], "uV")
        # The 100 uV tone dominates the raw span; the filtered span is the in-band
        # signal plus noise. Both are per-window half-ranges in uV, rounded up.
        self.assertGreater(payload["original_scale_hint_uv"], 100.0)
        self.assertLess(payload["original_scale_hint_uv"], 400.0)
        self.assertGreater(payload["filtered_scale_hint_uv"], 1.0)
        self.assertLess(payload["filtered_scale_hint_uv"], 60.0)


class TestNoAliasing(unittest.TestCase):
    """A captured display trace must survive later pushes unchanged.

    Mutation that makes this red: dropping the ``.copy()`` in
    ``AuditoryProcessor._capture_display``, which leaves ``window.raw`` aliasing
    storage the next ``feed`` reuses.
    """

    def test_the_ring_hands_out_copies_not_its_own_cells(self):
        """The invariant the chain's capture depends on, at the ring's own level.

        Mutation that makes this red: dropping the ``.copy()`` from
        :meth:`RawDisplayRing.tail`. ``tail`` is the only thing that turns the ring
        into a trace; if it returns a view, the next chunk's write lands in the same
        cells and rewrites a window that has already been published.
        """

        ring = RawDisplayRing(100)
        ring.push(np.arange(80.0))
        taken = ring.tail(60)
        self.assertFalse(
            np.shares_memory(taken, ring._ring),
            "the ring handed out a view of its own storage instead of a copy",
        )
        expected = taken.copy()
        ring.push(np.arange(1000.0, 1100.0))
        np.testing.assert_array_equal(taken, expected)

    def test_captured_display_is_immutable_across_a_later_push(self):
        # A short window on purpose: it makes the raw ring small, so one later
        # chunk can be made to wrap past the very cells the captured window came
        # from. With a 5 s window a chunk lands in fresh cells either way and the
        # aliasing this test exists to catch would go unnoticed.
        trial = SimpleNamespace(
            sample_rate=RATE, channel_names=CHANNELS, reference="fixture",
            upstream_processing="fixture",
        )
        built = stream_config(
            trial, AuditoryConfig(), history=3.0, step=1.0,
            check_channels=False, source_unit_exponent=-6, display_channel="Cz",
        )
        processor = AuditoryProcessor(built)
        eeg, time = trace(seconds=8.0, tone_hz=25.0)
        cut = 5 * int(RATE)
        windows = chain_windows(processor, eeg[:cut], time[:cut], block=4)
        self.assertTrue(windows, "the chain produced no window to capture")
        self.assertTrue(windows[-1].display, "the captured window carried no display")
        captured = windows[-1].display
        before_raw = np.asarray(windows[-1].raw, dtype=float).copy()
        before_samples = json.dumps(captured["samples"])

        # One more chunk, long enough to wrap the ring past those cells: the ring
        # holds window + 2 s here, and this push walks past its end.
        offset = cut
        wrapped = processor.feed(
            (eeg[offset : offset + 512], time[offset : offset + 512])
        )
        del wrapped  # only the reuse of the ring's storage matters here

        np.testing.assert_array_equal(np.asarray(windows[-1].raw, dtype=float), before_raw)
        self.assertEqual(json.dumps(captured["samples"]), before_samples)


class TestDecimationIsHonest(unittest.TestCase):
    """Lengths are computed, factors divide, and the decimation anti-aliases.

    Mutations that make this red: returning a bare ``values[::factor]`` slice
    (wrong length when the factor does not divide, and the alias test below); or
    choosing the factor by floor division instead of by the largest divisor.
    """

    def test_factor_divides_and_length_is_exactly_computed(self):
        cases = ((640, 8), (500, 5), (384, 4), (960, 12), (320, 4), (256, 4))
        for length, expected in cases:
            with self.subTest(length=length):
                factor = decimation_factor(length)
                self.assertEqual(factor, expected)
                self.assertEqual(length % factor, 0)
                self.assertEqual(len(decimate_trace(np.zeros(length), factor)), length // factor)

    def test_output_length_matches_the_windows_the_chain_really_builds(self):
        for history, expected_points in ((5.0, 80), (4.0, 64)):
            with self.subTest(history=history):
                trial = SimpleNamespace(
                    sample_rate=RATE, channel_names=CHANNELS, reference="fixture",
                    upstream_processing="fixture",
                )
                built = stream_config(
                    trial, AuditoryConfig(), history=history, step=1.0,
                    check_channels=False, source_unit_exponent=-6,
                    display_channel="Cz",
                )
                processor = AuditoryProcessor(built)
                eeg, time = trace(seconds=history + 2.0)
                window = chain_windows(processor, eeg, time, block=4)[-1]
                payload = window.display
                # Both traces reach the same computed point count, and that count
                # is the one the source lengths allow: never a slice that varies.
                self.assertEqual(len(payload["samples"]), expected_points)
                self.assertEqual(len(payload["filtered_samples"]), expected_points)
                self.assertEqual(
                    len(payload["samples"]),
                    len(window.raw) // shared_decimation(
                        len(window.raw), len(window.data)
                    )[0],
                )

    def test_decimation_anti_aliases_a_30hz_tone(self):
        """At 128 -> 16 Hz a 30 Hz tone aliases to 2 Hz; averaging must not keep it.

        Mutation that makes this red: ``decimate_for_display`` slicing with
        ``[::factor]`` instead of grouping and averaging.
        """

        time = np.arange(640) / RATE
        tone = np.sin(2 * np.pi * 30.0 * time)
        factor = decimation_factor(len(tone))
        averaged = decimate_trace(tone, factor)
        aliased = tone[::factor]
        rate = RATE / factor
        self.assertLess(float(np.max(np.abs(averaged))), 0.35)
        self.assertGreater(float(np.max(np.abs(aliased))), 0.9)
        self.assertLess(tone_amplitude(averaged, rate, 2.0), 0.2)
        self.assertGreater(tone_amplitude(aliased, rate, 2.0), 0.9)

    def test_a_factor_that_does_not_divide_is_refused(self):
        with self.assertRaises(ValueError):
            decimate_trace(np.zeros(100), 3)
        with self.assertRaises(ValueError):
            decimate_trace(np.zeros(640), 0)
        with self.assertRaises(ValueError):
            capture_display(
                np.zeros(10), np.zeros(0), channel="Cz", channel_source="test",
                source_rate=RATE, filtered_rate=OUTPUT_RATE, window_seconds=1.0,
                window_end=1.0, available_at=1.0, source_unit_exponent=-6,
            )


class TestPacketBudget(unittest.TestCase):
    """The frozen format, the real envelope, and the transport's size limit.

    Mutations that make this red: renaming or retyping a field in
    ``DisplayTap.to_payload``; swapping the two sample-rate fields; or making the
    decimation stop decimating, which grows the encoded size by the factor.
    """

    @classmethod
    def setUpClass(cls):
        eeg, time = trace(tone_hz=30.0)
        cls.processor = AuditoryProcessor(settings(display_channel="Cz"))
        cls.window = chain_windows(cls.processor, eeg, time, block=4)[-1]
        cls.payload = dict(cls.window.display)
        cls.payload["simulated"] = False

    def test_payload_types_match_the_frozen_format(self):
        payload = self.payload
        self.assertIsInstance(payload["sample_rate"], float)
        self.assertIsInstance(payload["filtered_sample_rate"], float)
        self.assertEqual(payload["channels"], ["Cz"])
        for trace in (payload["samples"], payload["filtered_samples"]):
            self.assertIsInstance(trace, list)
            self.assertTrue(trace)
            for row in trace:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), len(payload["channels"]))
                for value in row:
                    self.assertIsInstance(value, float)
                    self.assertTrue(np.isfinite(value))
        self.assertIsInstance(payload["original_units"], str)
        self.assertIsInstance(payload["filtered_units"], str)
        self.assertIsInstance(payload["original_scale_hint_uv"], float)
        self.assertIsInstance(payload["filtered_scale_hint_uv"], float)
        self.assertIsInstance(payload["window_seconds"], float)
        self.assertIsInstance(payload["lag_seconds"], float)
        self.assertIsInstance(payload["channel_source"], str)
        self.assertTrue(payload["channel_source"])
        self.assertEqual(set(payload) - {"simulated"}, PAYLOAD_FIELDS)

    def test_each_rate_describes_its_own_trace(self):
        payload = self.payload
        window_seconds = payload["window_seconds"]
        raw_drawn = len(payload["samples"]) / window_seconds
        filtered_drawn = len(payload["filtered_samples"]) / window_seconds
        # Each trace's own rate is the one that describes it: the two are drawn at
        # the same rate here, from different source rates, so a client that swapped
        # the two fields would draw the filtered trace at twice its real speed.
        self.assertAlmostEqual(raw_drawn, 16.0, places=6)
        self.assertAlmostEqual(filtered_drawn, 16.0, places=6)
        self.assertEqual(payload["sample_rate"], RATE)
        self.assertEqual(payload["filtered_sample_rate"], OUTPUT_RATE)
        self.assertNotEqual(payload["sample_rate"], payload["filtered_sample_rate"])

    def test_lag_is_finite_and_consistent_with_the_window_end(self):
        payload = self.payload
        self.assertTrue(np.isfinite(payload["lag_seconds"]))
        self.assertGreaterEqual(payload["lag_seconds"], 0.0)
        # The packet's own stamp is the newest raw sample's time, and the displayed
        # window ends no later than that: the lag is the distance between them, far
        # below one frame, not the window length.
        self.assertLess(payload["lag_seconds"], 0.05)
        self.assertAlmostEqual(
            payload["window_seconds"], len(self.window.timestamps) / OUTPUT_RATE
        )
        # The identity itself: the window end is exactly the stamp minus the lag.
        self.assertAlmostEqual(
            float(self.window.available_at) - payload["lag_seconds"],
            float(self.window.timestamps[-1]),
            places=9,
        )

    def test_encoded_packet_is_under_the_transport_limit(self):
        packet = make_packet(
            "eeg_display", 1.0, 1, "nova-aad", "session-test", self.payload
        )
        encoded = json.dumps(packet, separators=(",", ":")).encode()
        self.assertLess(len(encoded), MAX_PACKET_BYTES)
        # Room to spare: even undecimated this window would fit.
        self.assertLess(len(encoded) * 8, MAX_PACKET_BYTES)


class TestDefaultOffChangesNothing(unittest.TestCase):
    """No display channel means no capture, no field and no packet.

    Mutations that make this red: publishing ``eeg_display`` unconditionally; or
    defaulting ``RunPolicy.display_channel`` to ``"Cz"``.
    """

    def test_windows_carry_no_display_attribute(self):
        eeg, time = trace()
        processor = AuditoryProcessor(settings())
        self.assertIsNone(processor._display_channel)
        windows = chain_windows(processor, eeg, time)
        self.assertTrue(windows)
        for window in windows:
            self.assertFalse(hasattr(window, "raw"))
            self.assertFalse(hasattr(window, "display"))

    def test_default_policy_and_frame_carry_nothing(self):
        policy = RunPolicy(check_channels=False, max_bad_channels=0)
        self.assertIsNone(policy.display_channel)
        self.assertIsNone(policy.to_dict()["display_channel"])
        frame = frame_with({})
        self.assertEqual(frame.display, {})
        self.assertEqual(frame.to_dict()["display"], {})

    def test_no_packet_without_a_display_payload(self):
        producer, published = bare_producer()
        producer._emit_frame(lambda kind, *rest: published.append(kind), frame_with({}))
        self.assertNotIn("eeg_display", published)
        self.assertIn("attention", published)

    def test_a_payload_is_the_only_thing_that_publishes_it(self):
        producer, published = bare_producer()
        payload = {"sample_rate": RATE, "channels": ["Cz"]}
        producer._emit_frame(
            lambda kind, *rest: published.append(kind), frame_with(payload)
        )
        self.assertIn("eeg_display", published)

    def test_an_empty_display_channel_is_refused(self):
        with self.assertRaises(TypeError):
            RunPolicy(check_channels=False, max_bad_channels=0, display_channel="  ")


if __name__ == "__main__":
    unittest.main()
