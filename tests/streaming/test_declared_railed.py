"""A declared railed electrode stops itself, not the session.

The recorded ANT sessions rail two of their twenty electrodes (plan section
3.11: F8 at 83 333.3 uV for the whole of session 1, F3 for 39.8 % of it). Those
rails are *not* a chain bug, but they used to end the run: ``Repair`` bridges a
short damaged run by interpolating between two finite endpoints, and it refuses
to interpolate when an endpoint is railed past ``saturation_limit_uv`` (75 000 uV
by default) or when the two endpoints jump further than ``amplitude_limit_uv``
(500 uV). The repair span is the whole row, so a healthy electrode's damage
beside a railed one raised ``unsafe_endpoints`` and bounded recovery stopped the
session.

The fix is per-channel and already half-built: ``ChannelScope`` carries declared
electrodes into ``Repair``, which does not inspect them. What these tests pin
down is that it actually prevents the stop, that it does **not** blind the guard
for any other channel, and that an excluded electrode is still repaired, still
counted in the census, and still in the contract.

Everything here is on premises the test owns - synthetic microvolt traces at the
recording's own 500 Hz - so nothing depends on ``datasets/AAD-ANT`` being present.
"""

import json
import unittest
from types import SimpleNamespace

import numpy as np

from nova2026.auditory.session import RunPolicy
from nova2026.auditory.streaming import AuditoryProcessor, stream_config
from nova2026.streaming.preprocess import Repair
import scripts.getlive.ant_live as ant_live

RATE = 500.0
CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4",
    "T8", "Cz", "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",
)
# The rail the operator's amplifier produced, in uV (plan section 3.11: the
# 24-bit amplifier saturates at +-83.9 mV).
RAIL_UV = 83333.3
# Where the declared electrode and a healthy one sit in the column order.
RAILED = CHANNELS.index("F8")
HEALTHY = CHANNELS.index("F3")


def railed_eeg(samples=3000, *, railed=RAILED, damaged=HEALTHY, damage=(50, 56),
               extra_damage=None, amplitude=60.0):
    """A 500 Hz trace: one electrode railed, one short NaN run on another.

    The NaN run is the damage ``Repair`` must bridge, and it sits *after* the
    first finite row so the bridge is a real interpolation rather than the
    "damage before the run started" drop. ``extra_damage`` is ``(column,
    (first, last))`` for a second damaged electrode, which is how a test keeps an
    in-scope electrode carrying damage while another is declared. Returns the
    microvolt array and the sample timestamps.
    """

    time = np.arange(samples, dtype=np.float64) / RATE
    eeg = amplitude * np.sin(2 * np.pi * 7.0 * time)[:, None] * np.ones((1, len(CHANNELS)))
    eeg[:, railed] = RAIL_UV
    first, last = damage
    eeg[first:last, damaged] = np.nan
    if extra_damage is not None:
        column, (first, last) = extra_damage
        eeg[first:last, column] = np.nan
    return eeg, time


def settings(*, exclude=(), exponent=-6, rate=RATE, **kwargs):
    """The chain settings the live runner builds, as a plain namespace."""

    trial = SimpleNamespace(
        sample_rate=rate, channel_names=CHANNELS, reference="CPz",
        upstream_processing="unit test fixture",
    )
    config = SimpleNamespace(sample_rate=128.0, band=(0.5, 32.0), lag_samples=10)
    return stream_config(
        trial, config, history=5.0, step=1.0, check_channels=False,
        exclude_channels=exclude, source_unit_exponent=exponent, **kwargs
    )


def drive(processor, eeg, time, block=100):
    """Feed the whole trace, returning ``(windows, kind_of_stop_or_None, repairs)``.

    The chain does not raise ``UnrepairableError`` out of ``feed``: bounded
    recovery catches it, resets the stateful stages, counts the recovery and
    carries on. That is the live behaviour, and it is why the live ANT replay
    stopped *late* rather than at the first railed bridge: it took as many
    unsafe bridges as the recovery budget allowed. This helper therefore counts
    the recoveries and asks whether the guard would still be allowing another
    one - the state the next chunk would stop in - instead of feeding a long
    trace to provoke the same error five more times.
    """

    windows = []
    for start in range(0, len(eeg), block):
        try:
            windows.extend(processor.feed((eeg[start : start + block],
                                           time[start : start + block])))
        except RuntimeError:
            return windows, processor.recovery.events[-1]["kind"], \
                processor.repair.repaired_samples
        recovery = processor.recovery
        if recovery.recoveries >= recovery.max_events and recovery.events:
            # The budget is spent: the next unrepairable chunk raises, which is
            # exactly how the recorded ANT replay ended. Report the state the
            # run is in rather than feeding a long trace to provoke it.
            return windows, recovery.events[-1]["kind"], \
                processor.repair.repaired_samples
    return windows, None, processor.repair.repaired_samples


def recovery_history(eeg, time, *, exclude=(), passes=8):
    """Replay the trace and report every fault bounded recovery was asked to absorb.

    One unsafe bridge costs one recovery (the stages are reset and the run
    continues), so the trace has to be replayed to spend the budget - which is
    what actually ended the recorded ANT replay. Returns ``(recoveries, kinds,
    repaired_samples, stopped)``: the count, the fault names in order, how many
    damaged samples were bridged at all, and whether the guard finally refused
    to recover (the run-stopping error).
    """

    processor = AuditoryProcessor(settings(exclude=exclude))
    span = float(time[-1] - time[0]) + 1.0 / RATE
    stopped = False
    for attempt in range(passes):
        try:
            for start in range(0, len(eeg), 100):
                stamp = time[start : start + 100] + attempt * span
                processor.feed((eeg[start : start + 100], stamp))
        except RuntimeError:
            stopped = True
            break
    return (
        processor.recovery.recoveries,
        [event["kind"] for event in processor.recovery.events],
        processor.repair.repaired_samples,
        stopped,
    )


class ExclusionPreventsTheStopTests(unittest.TestCase):
    """A declared railed electrode cannot end the session; nothing else moves."""

    def test_a_railed_electrode_beside_short_damage_stops_the_run_when_undeclared(self):
        # The premise, on the same fixture the next test uses: this is what the
        # live ANT replay did - one unsafe bridge costs one recovery, the
        # stateful stages are reset, and the budget is spent one bridge at a time
        # until there is none left and the run stops.
        processor = AuditoryProcessor(settings())
        eeg, time = railed_eeg()
        _, stopped, repairs = drive(processor, eeg, time)
        self.assertEqual(processor.recovery.max_events, 5)
        self.assertEqual(processor.recovery.recoveries, 1)
        self.assertEqual(processor.recovery.segment, 1)
        self.assertEqual(processor.recovery.events[0]["kind"], "unsafe_endpoints")
        self.assertEqual(repairs, 0, "an unsafe bridge was repaired anyway")
        recoveries, kinds, _, stopped = recovery_history(eeg, time)
        self.assertTrue(stopped, "the recovery budget never ran out")
        # `Recovery.handle` counts the recovery it is refusing, so the counter
        # reads one past the budget on the stop itself.
        self.assertGreaterEqual(recoveries, processor.recovery.max_events)
        self.assertEqual(set(kinds), {"unsafe_endpoints"})

    def test_declaring_it_excluded_lets_the_session_run_on(self):
        processor = AuditoryProcessor(settings(exclude=("F8",)))
        eeg, time = railed_eeg()
        windows, stopped, repairs = drive(processor, eeg, time)
        self.assertIsNone(stopped)
        self.assertEqual(processor.recovery.recoveries, 0)
        self.assertEqual(repairs, 6, "the healthy electrode's damage went unrepaired")
        self.assertGreater(len(windows), 0, "the chain produced no window at all")
        # ... and it keeps running: replaying the same trace never spends a
        # budget it never touches.
        recoveries, kinds, _, _ = recovery_history(eeg, time, exclude=("F8",))
        self.assertEqual((recoveries, kinds), (0, []))

    def test_the_excluded_electrode_is_still_counted_and_keeps_its_column(self):
        # Nothing may silently drop a column: the excluded electrode keeps its
        # data, damage on it is handled (held at its last finite level, since an
        # excluded electrode may not wait for a right endpoint it may never get)
        # and the quality census still names it - the census is evidence for the
        # run record, not a verdict.
        processor = AuditoryProcessor(settings(exclude=("F8",)))
        eeg, time = railed_eeg(railed=RAILED, damaged=RAILED, damage=(50, 56))
        windows, kind, _ = drive(processor, eeg, time)
        self.assertIsNone(kind)
        self.assertEqual(processor.repair.held_rows, 6)
        self.assertEqual(processor.repair.exclude_channels, ("F8",))
        self.assertIn("F8", processor.quality.channel_names)
        census = set(processor.quality.bad_channels(0.0, time[-1]))
        self.assertIn("F8", census, "the excluded electrode left the census")
        self.assertTrue(windows)
        self.assertEqual(windows[-1].data.shape[1], len(CHANNELS))
        # The held rows carry that electrode's LAST FINITE LEVEL: not a hole
        # (which would poison every causal filter downstream) and not a zero
        # (which would invent a step the amplifier never produced).
        repair = Repair(
            RATE, source_unit_exponent=-6, channel_names=CHANNELS,
            exclude_channels=("F8",),
        )
        repaired, _ = repair(eeg[:100], time[:100])
        self.assertEqual(repair.held_rows, 6)
        rail_column = repaired[:, RAILED]
        self.assertTrue(np.isfinite(rail_column).all())
        np.testing.assert_allclose(rail_column[50:56], RAIL_UV, rtol=0, atol=1e-9)
        # Damage before the first finite row has no level to hold: the column
        # starts at zero rather than staying NaN.
        early, early_time = railed_eeg(railed=RAILED, damaged=RAILED, damage=(0, 4))
        repair = Repair(
            RATE, source_unit_exponent=-6, channel_names=CHANNELS,
            exclude_channels=("F8",),
        )
        rows, _ = repair(early[:100], early_time[:100])
        self.assertEqual(repair.held_rows, 4)
        np.testing.assert_array_equal(rows[:4, RAILED], np.zeros(4))

    def test_the_healthy_electrodes_damage_beside_it_is_still_repaired(self):
        # The exclusion is per-channel in both directions: declaring F8 must not
        # stop F3's damage from being repaired normally.
        processor = AuditoryProcessor(settings(exclude=("F8",)))
        eeg, time = railed_eeg(railed=RAILED, damaged=HEALTHY, damage=(50, 56))
        windows, kind, repairs = drive(processor, eeg, time)
        self.assertIsNone(kind)
        self.assertEqual(repairs, 6)
        self.assertEqual(processor.repair.held_rows, 0)
        self.assertTrue(windows)

    def test_a_railed_electrode_inside_the_guard_still_stops_the_run(self):
        # The exclusion is checked against a measured rail, not trusted: a
        # declared electrode BELOW the guard is not railed and must not be
        # excludable silently, or the operator could hide real damage.
        from scripts.getlive.ant_live import check_excluded_channels, rail_profile

        eeg, _ = railed_eeg(railed=RAILED)
        # 6 mV: an electrode with real excursions, but nowhere near the 75 000 uV
        # rail and not held at any level for long.
        eeg[:, RAILED] = 6000.0 * np.sin(2 * np.pi * 3.0 * np.arange(len(eeg)) / RATE)
        profile = rail_profile(eeg, CHANNELS)
        self.assertNotIn("F8", profile["railed_channels"])
        with self.assertRaises(ValueError) as caught:
            check_excluded_channels(
                profile, ("F8",), saturation_limit_uv=75000.0, allow_unrailed=False
            )
        self.assertIn("not railed", str(caught.exception))
        # ... and the explicit override is the only way through, and it says so.
        allowed = check_excluded_channels(
            profile, ("F8",), saturation_limit_uv=75000.0, allow_unrailed=True
        )
        self.assertTrue(any("NOT railed" in line for line in allowed))


class OtherChannelsKeepTheirProtectionTests(unittest.TestCase):
    """The exclusion is per-channel: every other electrode keeps the full guard."""

    def test_a_railed_electrode_that_is_not_excluded_still_stops_the_run(self):
        # The case a wider global limit would have blinded. F8 is railed, F3 is
        # declared (and carries damage of its own) and Pz carries the bridge that
        # forces the check. Declaring F3 must leave F8 under the guard: the
        # bridge is still refused and the run still spends its budget.
        third = CHANNELS.index("Pz")
        eeg, time = railed_eeg(
            railed=RAILED, damaged=HEALTHY, damage=(50, 56),
            extra_damage=(third, (150, 156)),
        )
        recoveries, kinds, repaired, _ = recovery_history(eeg, time, exclude=("F3",))
        self.assertGreaterEqual(recoveries, 1)
        self.assertEqual(set(kinds), {"unsafe_endpoints"})
        self.assertEqual(repaired, 0)
        # Declaring the railed electrode instead is what lets the same trace run
        # on, with the other electrodes' damage repaired normally.
        declared = AuditoryProcessor(settings(exclude=("F8",)))
        windows, stopped, repaired = drive(declared, eeg, time)
        self.assertIsNone(stopped)
        self.assertEqual(declared.recovery.recoveries, 0)
        self.assertEqual(repaired, 12)
        self.assertTrue(windows)

    def test_declaring_them_both_still_records_both(self):
        # Two declarations, and the record still names the rail of each: the
        # exclusion is evidence-preserving, not a blanket silence.
        from nova2026.streaming.preprocess import Repair

        repair = Repair(
            RATE, source_unit_exponent=-6, channel_names=CHANNELS,
            exclude_channels=("F8", "F3"),
        )
        self.assertEqual(repair.scope.excluded_indices,
                         frozenset({CHANNELS.index("F8"), CHANNELS.index("F3")}))
        processor = AuditoryProcessor(settings(exclude=("F8", "F3")))
        self.assertEqual(processor.quality.exclude_channels, ("F8", "F3"))
        eeg, time = railed_eeg(railed=RAILED, damaged=HEALTHY, damage=(50, 56))
        _, stopped, _ = drive(processor, eeg, time)
        self.assertIsNone(stopped)
        # The railed electrode is still reported: exclusion decides what may stop
        # the run, never what may be recorded.
        census = set(processor.quality.bad_channels(0.0, time[-1]))
        self.assertIn("F8", census)

    def test_the_guard_keeps_its_own_value_for_every_other_channel(self):
        # Repair's limits are untouched by an exclusion: the excluded channel is
        # skipped in the check, not removed from it, and the numbers stay the
        # chain's own defaults (500 uV jump / 75 000 uV rail).
        plain = Repair(RATE, source_unit_exponent=-6, channel_names=CHANNELS)
        declared = Repair(
            RATE, source_unit_exponent=-6, channel_names=CHANNELS,
            exclude_channels=("F8",),
        )
        for repair in (plain, declared):
            self.assertEqual(repair._amplitude_uv, 500.0)
            self.assertEqual(repair._saturation_uv, 75000.0)
            self.assertEqual(repair.channel_names, CHANNELS)
        self.assertEqual(declared.scope.mask(len(CHANNELS)).sum(), len(CHANNELS) - 1)
        self.assertTrue(plain.scope.mask(len(CHANNELS)).all())

    def test_an_unknown_exclusion_is_refused_rather_than_ignored(self):
        with self.assertRaises(ValueError) as caught:
            AuditoryProcessor(settings(exclude=("FCz",)))
        self.assertIn("FCz", str(caught.exception))


class DefaultsAreUnchangedTests(unittest.TestCase):
    """A session that excludes nothing behaves exactly as before."""

    def test_the_contract_gains_no_key_for_the_channel_policy(self):
        # The contract is compared against a trained model's contract verbatim,
        # so a new key would invalidate every existing decoder. The channel
        # policy rides in the run policy instead.
        plain = AuditoryProcessor(settings())
        declared = AuditoryProcessor(settings(exclude=("F8",)))
        self.assertEqual(plain.contract, declared.contract)
        self.assertNotIn("exclude_channels", declared.contract)
        self.assertNotIn("saturation_limit_uv", declared.contract)

    def test_the_run_policy_records_the_exclusion_without_touching_the_contract(self):
        policy = RunPolicy(
            check_channels=False, max_bad_channels=0, exclude_channels=("F8", "F3")
        )
        record = policy.to_dict()
        self.assertEqual(record["exclude_channels"], ["F8", "F3"])

    def test_no_declaration_means_every_column_is_in_scope(self):
        processor = AuditoryProcessor(settings())
        self.assertEqual(processor.repair.exclude_channels, ())
        self.assertEqual(processor.quality.exclude_channels, ())
        self.assertTrue(processor.repair.scope.mask(len(CHANNELS)).all())

    def test_a_clean_trace_is_untouched_by_the_exclusion_path(self):
        # The same clean trace through a plain chain and a chain with a declared
        # electrode must produce identical windows: declaring a dead electrode
        # changes no sample of a healthy recording.
        eeg, time = railed_eeg()
        eeg[:, RAILED] = 25.0  # nothing railed in this trace
        clean, _, _ = drive(AuditoryProcessor(settings()), eeg, time)
        declared, _, _ = drive(AuditoryProcessor(settings(exclude=("F8",))), eeg, time)
        self.assertEqual(len(clean), len(declared))
        self.assertGreater(len(clean), 0)
        for left, right in zip(clean, declared):
            np.testing.assert_array_equal(left.data, right.data)
            self.assertEqual(left.reasons, right.reasons)

    def test_a_declared_exclusion_is_a_label_not_a_column_number(self):
        # A column index would silently exclude the wrong electrode on a rig
        # that delivers its columns in another order.
        processor = AuditoryProcessor(settings(exclude=("F8",)))
        self.assertEqual(processor.repair.scope.excluded_indices,
                         frozenset({CHANNELS.index("F8")}))


class RailProfileTests(unittest.TestCase):
    """The declaration's evidence is measured on the published window."""

    def test_the_operator_sessions_two_railed_electrodes_are_named(self):
        from scripts.getlive.ant_live import rail_profile

        eeg, _ = railed_eeg(samples=2000)
        eeg[:1000, HEALTHY] = RAIL_UV  # F3 railed for half the window, as measured
        profile = rail_profile(eeg, CHANNELS)
        self.assertEqual(sorted(profile["railed_channels"]), ["F3", "F8"])
        self.assertAlmostEqual(
            profile["per_channel"]["F8"]["railed_fraction"], 1.0, places=6
        )
        self.assertAlmostEqual(
            profile["per_channel"]["F3"]["railed_fraction"], 0.5, places=6
        )
        self.assertTrue(all(not row["railed"] for row in
                            (profile["per_channel"][name] for name in ("Fp1", "Cz", "Oz"))))

    def test_a_single_spike_is_not_a_rail(self):
        from scripts.getlive.ant_live import rail_profile

        eeg, _ = railed_eeg()
        eeg[:, RAILED] = 30.0
        eeg[700, RAILED] = RAIL_UV
        profile = rail_profile(eeg, CHANNELS)
        self.assertNotIn("F8", profile["railed_channels"])

    def test_the_runner_refuses_a_label_that_is_not_an_electrode(self):
        from scripts.getlive.ant_live import check_excluded_channels, rail_profile

        profile = rail_profile(railed_eeg()[0], CHANNELS)
        with self.assertRaises(ValueError) as caught:
            check_excluded_channels(
                profile, ("FCz",), saturation_limit_uv=75000.0, allow_unrailed=False
            )
        self.assertIn("not one of the 20 published electrodes", str(caught.exception))


class RunnerPolicyWiringTests(unittest.TestCase):
    """The runner's defaults are policy, and the defaults are the old behaviour."""

    def test_the_saturation_guard_defaults_to_the_chain_own_limit(self):
        from scripts.getlive.ant_live import (
            REPAIR_DEFAULT_SATURATION_UV,
            build_parser,
            saturation_guard_uv,
        )

        args = build_parser().parse_args([])
        self.assertIsNone(args.saturation_limit_uv)
        self.assertEqual(args.exclude_channels, ())
        self.assertFalse(args.allow_unrailed_exclusion)
        self.assertEqual(saturation_guard_uv(args), REPAIR_DEFAULT_SATURATION_UV)
        # The runner's idea of the chain default must BE the chain default: a
        # copy that drifts upward is the rejected design arriving by accident.
        self.assertEqual(REPAIR_DEFAULT_SATURATION_UV, 75000.0)
        from nova2026.auditory.streaming import AuditoryProcessor

        chain = AuditoryProcessor(settings())
        self.assertEqual(chain.repair._saturation_uv, REPAIR_DEFAULT_SATURATION_UV)

    def test_excluded_channels_are_parsed_as_labels(self):
        from scripts.getlive.ant_live import build_parser

        args = build_parser().parse_args(["--exclude-channels", "F8, F3 ,"])
        self.assertEqual(args.exclude_channels, ("F8", "F3"))

    def test_the_audio_offset_stays_the_operator_given_one(self):
        from scripts.getlive.ant_live import build_parser

        args = build_parser().parse_args(["--audio-start-offset", "267"])
        self.assertEqual(args.audio_start_offset, 267.0)
        self.assertIn("3.12", build_parser().format_help())

    def test_the_decision_census_separates_absence_from_abstention(self):
        from scripts.getlive.ant_live import decision_census

        summary = {
            "windows": 10, "evidence_gaps": 2,
            "decisions": {"A": 3, "B": 2, "uncertain": 2, "unavailable": 1},
        }
        census = decision_census(summary, [(1.0, "A"), (2.0, "B")])
        self.assertEqual(census["decided"], 5)
        self.assertEqual(census["A"], 3)
        self.assertEqual(census["B"], 2)
        self.assertEqual(census["uncertain"], 2)
        self.assertEqual(census["unavailable"], 1)
        self.assertEqual(census["evidence_gap"], 2)
        self.assertEqual(census["decision_stream"], [[1.0, "A"], [2.0, "B"]])

    def test_the_operator_rail_is_measured_and_recorded_on_the_session_file(self):
        # The premises of the ANT replay, asserted without the recording: F8 and
        # F3 select the same columns the runner would exclude by name, and the
        # policy record carries the labels rather than column numbers.
        policy = RunPolicy(
            check_channels=False, max_bad_channels=0, exclude_channels=("F8", "F3")
        )
        self.assertEqual(
            [CHANNELS.index(name) for name in policy.exclude_channels], [6, 3]
        )
        self.assertEqual(policy.to_dict()["exclude_channels"], ["F8", "F3"])


class DeclaredRateTests(unittest.TestCase):
    """The chain declares the rate it is fed, not the rate of the outlet.

    The pre-chain adapter (500 Hz outlet -> the 128 Hz the decoder contract
    records) changes the data the chain sees. If the declaration does not move
    with it, `Repair` reads one 128 Hz step as 3.9 samples at 500 Hz, invents
    three missing rows per step and rejects every window as `interpolated` - the
    failure that actually ended the live ANT replay (7608 phantom repairs)
    before the railed electrodes were reached.
    """

    def trace(self, declared_rate, samples=12800):
        rng = np.random.default_rng(0)
        eeg = rng.normal(0.0, 20.0, (samples, len(CHANNELS)))
        stamps = np.arange(samples) / 128.0 + 5.0
        processor = AuditoryProcessor(settings(rate=declared_rate))
        windows, stopped = [], None
        try:
            for start in range(0, samples, 100):
                windows.extend(
                    processor.feed((eeg[start : start + 100], stamps[start : start + 100]))
                )
        except RuntimeError as error:
            stopped = str(error)
        return processor, windows, stopped

    def test_the_rate_the_chain_declares_is_the_rate_it_is_fed(self):
        processor, windows, stopped = self.trace(128.0)
        self.assertEqual(processor.contract["input_sfreq"], 128.0)
        self.assertEqual(processor.resampler.quality, "pass-through")
        self.assertIsNone(stopped)
        self.assertGreater(len(windows), 50)
        self.assertEqual({tuple(window.reasons) for window in windows}, {()})
        self.assertEqual(processor.repair.repaired_samples, 0)

    def test_a_stale_declaration_invents_repairs_and_stops_the_run(self):
        processor, windows, stopped = self.trace(500.0)
        self.assertEqual(processor.contract["input_sfreq"], 500.0)
        self.assertGreater(processor.repair.repaired_samples, 100)
        self.assertTrue(all("interpolated" in window.reasons for window in windows))
        self.assertIsNotNone(stopped)
        self.assertIn("persisted beyond the allowed duration", stopped)

    def test_the_runner_declares_the_rate_the_adapter_delivers(self):
        pre = SimpleNamespace(pre_resample="auto", sfreq=500.0)
        model = SimpleNamespace(contract={"input_sfreq": 128.0})
        self.assertEqual(ant_live.chain_rate(pre, model), 128.0)
        off = SimpleNamespace(pre_resample="off", sfreq=500.0)
        self.assertEqual(ant_live.chain_rate(off), 500.0)


class SessionPolicyWiringTests(unittest.TestCase):
    """The runner's declaration reaches the session it builds."""

    def test_build_hands_the_declared_electrodes_and_rate_to_the_session(self):
        captured = {}

        class StubSession:
            def __init__(self, *, source, decoder, references, policy):
                captured["policy"] = policy
                captured["rate"] = source.sample_rate

        class StubProducer:
            def __init__(self, session):
                captured["producer_session"] = session

        original_session = ant_live.AttentionSession
        original_producer = ant_live.AttentionProducer
        ant_live.AttentionSession = StubSession
        ant_live.AttentionProducer = StubProducer
        self.addCleanup(setattr, ant_live, "AttentionSession", original_session)
        self.addCleanup(setattr, ant_live, "AttentionProducer", original_producer)

        args = ant_live.build_parser().parse_args(
            ["--exclude-channels", "F8,F3", "--saturation-limit-uv", "60000"]
        )
        source = SimpleNamespace(sample_rate=500.0, audio_start=0.0)
        ant_live.build(
            {
                "source": source,
                "references": None,
                "model": None,
                "media": None,
                "audio_first_timestamp": 5.0,
                "audio_offset": 0.0,
                "excluded": ("F8", "F3"),
                "chain_rate": 128.0,
            },
            args,
        )
        policy = captured["policy"]
        self.assertEqual(policy.exclude_channels, ("F8", "F3"))
        self.assertEqual(policy.saturation_limit_uv, 60000.0)
        self.assertEqual(captured["rate"], 128.0)
        self.assertEqual(source.sample_rate, 128.0)


if __name__ == "__main__":  # pragma: no cover - convenience
    unittest.main()
