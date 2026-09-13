"""The quality limit is declared and measured, never assumed.

The operator's ANT session moves 1.5-3.0 mV from its own anchor level on 18 of
its 20 electrodes, against a chain whose ``QualityMonitor`` calls an electrode
faulted once it moves further than 500 uV. At that default every window is an
artifact, ``signal_quality`` is 0 and no decision is ever committed: measured,
121 windows, 0 decided. The fix is the same shape as the declared railed
electrode (decision D-37): make the limit a **declared run-policy value**, keep
the chain's default exactly where it is, and record the value in force.

What these tests pin down:

* the declared value reaches the one owner of the decision - both stages that
  judge an electrode with it, ``QualityMonitor`` and ``Repair`` - so the quality
  verdict and the repair verdict cannot disagree;
* the default is unchanged, and unchanged **because nothing sets it**, which is
  proved by reading the chain's own attributes rather than by asserting a
  constant that a future edit could move together with the code;
* declaring a limit does not silently disable the other quality criteria:
  saturation and flatline still fault, and ``check_channels`` still decides;
* a declared limit still refuses what a declared limit should not admit.

Everything here is on premises the test owns - synthetic microvolt traces at the
recording's own 500 Hz - so nothing depends on ``datasets/AAD-ANT`` being
present.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nova2026.auditory.session import AttentionSession, RunPolicy
from nova2026.auditory.streaming import AuditoryProcessor, stream_config
from nova2026.streaming.preprocess import QualityMonitor, Repair
import scripts.getlive.ant_live as ant_live
from nova2026.auditory.decoder import RidgeDecoder

REPO = Path(__file__).resolve().parents[2]
RATE = 500.0
CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4",
    "T8", "Cz", "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2",
)
# The rail the operator's amplifier produced, in uV (plan section 3.11), and the
# excursion the operator's own recording reaches on its usable electrodes: the
# declared limit below sits above the second and far below the first, which is the
# whole argument for declaring a number instead of editing a default.
RAIL_UV = 83333.3
DECLARED_UV = 20000.0
"""The value the ANT run declares, measured at the point the monitor is fed.

Nothing is tuned here. The chain's monitor sees the resampled 128 Hz stream, and
on that stream this recording's largest excursion on a usable electrode is
19 522 uV - five times the 3 637 uV the raw 500 Hz window shows, because the
pre-chain resampler's own transient moves the anchor row. The first declared run
used the raw number and left eight electrodes faulted; the declared value is the
measured one with a margin, and the run record carries the measurement.
"""
# A single electrode spike no declared limit should ever admit as normal signal.
SPIKE_UV = 100000.0
"""Chosen above the declared limit, because the declared limit is a large number.

A 10 000 uV spike would be *inside* a 20 000 uV declaration and admitting it would
be correct behaviour, so the "still refuses" case has to be stated in units this
declaration actually distinguishes.
"""


def trace(samples=3000, *, amplitude=60.0, drift_uv=2500.0):
    """A 500 Hz trace with a slow excursion on one electrode.

    The excursion is a ramp rather than a sinusoid so it is unmistakably the
    "moved away from the anchor and stayed there" case the monitor's amplitude
    fault describes, not a phase of an oscillation it would recover from.
    """

    time = np.arange(samples, dtype=np.float64) / RATE
    eeg = amplitude * np.sin(2 * np.pi * 7.0 * time)[:, None] * np.ones((1, len(CHANNELS)))
    eeg[:, :] += np.linspace(0.0, drift_uv, samples)[:, None]
    return eeg, time


def settings(*, exclude=(), rate=RATE, **kwargs):
    """The chain settings the live runner builds, as a plain namespace."""

    trial = SimpleNamespace(
        sample_rate=rate, channel_names=CHANNELS, reference="CPz",
        upstream_processing="unit test fixture",
    )
    config = SimpleNamespace(sample_rate=128.0, band=(0.5, 32.0), lag_samples=10)
    return stream_config(
        trial, config, history=5.0, step=1.0,
        check_channels=bool(kwargs.pop("check_channels", False)),
        exclude_channels=exclude, source_unit_exponent=-6, **kwargs
    )


def fit(eeg, time, *, block=100, **kwargs):
    """Feed one trace through a real chain; return it and every window reason."""

    processor = AuditoryProcessor(settings(**kwargs))
    windows = []
    for start in range(0, len(eeg), block):
        windows.extend(processor.feed((eeg[start : start + block],
                                       time[start : start + block])))
    return processor, windows


class PointingAtOneOwnerTests(unittest.TestCase):
    """The declared value reaches both stages that judge an electrode with it."""

    def test_a_declared_limit_reaches_the_monitor_and_the_repairer(self):
        processor = AuditoryProcessor(settings(amplitude_limit_uv=DECLARED_UV))
        self.assertEqual(processor.quality._amplitude_limit, DECLARED_UV)
        self.assertEqual(processor.repair._amplitude_uv, DECLARED_UV)

    def test_the_saturation_guard_is_not_dragged_along(self):
        # The two limits are separate decisions: declaring an excursion limit must
        # not move the absolute rail, which is what names a railed electrode.
        processor = AuditoryProcessor(settings(amplitude_limit_uv=DECLARED_UV))
        self.assertEqual(processor.quality._saturation_limit,
                         ant_live.REPAIR_DEFAULT_SATURATION_UV)
        self.assertEqual(processor.repair._saturation_uv,
                         ant_live.REPAIR_DEFAULT_SATURATION_UV)

    def test_the_run_policy_records_the_value_it_declared(self):
        policy = RunPolicy(
            check_channels=False, max_bad_channels=0,
            amplitude_limit_uv=DECLARED_UV, exclude_channels=("F8", "F3"),
        )
        record = policy.to_dict()
        self.assertEqual(record["amplitude_limit_uv"], DECLARED_UV)
        self.assertEqual(record["exclude_channels"], ["F8", "F3"])

    def test_the_contract_gains_no_key_for_the_declared_limit(self):
        # The contract is compared against a trained model's contract verbatim,
        # so a new key would invalidate every existing decoder.
        plain = AuditoryProcessor(settings())
        declared = AuditoryProcessor(settings(amplitude_limit_uv=DECLARED_UV))
        self.assertEqual(plain.contract, declared.contract)
        self.assertNotIn("amplitude_limit_uv", declared.contract)


class DefaultIsUnchangedTests(unittest.TestCase):
    """A caller that declares nothing gets exactly what it got before."""

    def test_the_chain_defaults_are_read_off_the_chain_not_restated(self):
        for processor in (AuditoryProcessor(settings()), QualityMonitor(
            n_eeg=len(CHANNELS), sfreq=RATE
        ), Repair(RATE, source_unit_exponent=-6)):
            if isinstance(processor, QualityMonitor):
                self.assertEqual(processor._amplitude_limit, 500.0)
                self.assertEqual(processor._saturation_limit, 75000.0)
            elif isinstance(processor, Repair):
                self.assertEqual(processor._amplitude_uv, 500.0)
                self.assertEqual(processor._saturation_uv, 75000.0)
            else:
                self.assertEqual(processor.quality._amplitude_limit, 500.0)
                self.assertEqual(processor.repair._amplitude_uv, 500.0)
                self.assertEqual(processor.quality._saturation_limit, 75000.0)

    def test_no_declaration_means_the_policy_field_is_none(self):
        # ``None`` is "the chain decides", not a copy of the default: a value
        # copied here would move the day the chain's default moves.
        policy = RunPolicy(check_channels=False, max_bad_channels=0)
        self.assertIsNone(policy.amplitude_limit_uv)
        self.assertIsNone(policy.to_dict()["amplitude_limit_uv"])

    def test_the_runner_declares_nothing_unless_asked(self):
        args = ant_live.build_parser().parse_args([])
        self.assertIsNone(args.amplitude_limit_uv)
        self.assertEqual(ant_live.amplitude_guard_uv(args), 500.0)
        self.assertEqual(ant_live.amplitude_guard_uv(args),
                         ant_live.REPAIR_DEFAULT_AMPLITUDE_UV)

    def test_a_declared_limit_is_the_runner_s_own_number(self):
        args = ant_live.build_parser().parse_args(
            ["--amplitude-limit-uv", str(DECLARED_UV)]
        )
        self.assertEqual(ant_live.amplitude_guard_uv(args), DECLARED_UV)

    def test_the_same_trace_is_unchanged_when_nothing_is_declared(self):
        # The declaration path must be inert for a caller that does not use it:
        # the same samples through a chain that was told nothing and a chain told
        # the default explicitly are the same run.
        eeg, time = trace(drift_uv=200.0)
        silent, windows_silent = fit(eeg, time)
        explicit, windows_explicit = fit(eeg, time,
                                         amplitude_limit_uv=500.0)
        self.assertEqual(len(windows_silent), len(windows_explicit))
        for left, right in zip(windows_silent, windows_explicit):
            np.testing.assert_array_equal(left.data, right.data)
            self.assertEqual(left.reasons, right.reasons)
            self.assertEqual(silent.quality._amplitude_limit,
                             explicit.quality._amplitude_limit)


class OtherCriteriaSurviveTests(unittest.TestCase):
    """Declaring the excursion limit moves one criterion and no other."""

    def test_a_drifting_recording_faults_at_the_default_and_not_at_the_declared(self):
        # This is the measured ANT case in miniature: a real recording's own
        # movement, judged by the chain's default and by a declared limit. The
        # amplitude fault is a *channel* fault, so under the relaxed policy it is
        # read from the census (`fault_channels`) rather than from `reasons` -
        # which is exactly the path the session's `signal_quality` verdict uses.
        eeg, time = trace(drift_uv=2500.0)
        at_default, windows_default = fit(eeg, time)
        declared, windows_declared = fit(eeg, time,
                                         amplitude_limit_uv=DECLARED_UV)
        band = (time[0], time[-1])
        self.assertIn("amplitude", at_default.quality.fault_channels(*band))
        self.assertEqual(declared.quality.fault_channels(*band), {})
        self.assertGreater(len(windows_default), 0)
        for window in windows_default:
            self.assertTrue(window.bad_channels)
        for window in windows_declared:
            self.assertEqual(window.bad_channels, ())
            self.assertEqual(window.reasons, ())

    def test_a_railed_electrode_is_still_faulted_at_the_declared_limit(self):
        # Saturation is an absolute rail, so declaring an excursion limit must not
        # absorb it: the electrode that railed is still named.
        eeg, time = trace(drift_uv=100.0)
        eeg[:, CHANNELS.index("F8")] = RAIL_UV
        processor, _ = fit(eeg, time, amplitude_limit_uv=DECLARED_UV)
        band = (time[0], time[-1])
        self.assertIn("saturation", processor.quality.fault_channels(*band))
        self.assertIn("F8", processor.quality.bad_channels(*band))

    def test_a_flat_electrode_is_still_faulted_at_the_declared_limit(self):
        # Flatline is a different fault with its own criterion; a declared
        # amplitude limit must not quietly switch it off.
        eeg, time = trace(drift_uv=100.0)
        eeg[:, CHANNELS.index("Oz")] = 12.0
        processor, _ = fit(eeg, time, amplitude_limit_uv=DECLARED_UV)
        band = (time[0], time[-1])
        self.assertIn("flatline", processor.quality.fault_channels(*band))

    def test_check_channels_still_decides_at_a_declared_limit(self):
        # The two policies are independent: a declared limit with the channel
        # check on still rejects, which is what makes the declaration a limit and
        # not a mute.
        eeg, time = trace(drift_uv=100.0)
        eeg[:, CHANNELS.index("F8")] = RAIL_UV
        processor = AuditoryProcessor(
            settings(amplitude_limit_uv=DECLARED_UV, check_channels=True)
        )
        for start in range(0, len(eeg), 100):
            processor.feed((eeg[start : start + 100], time[start : start + 100]))
        self.assertIn("saturation", processor.quality.reasons(time[0], time[-1]))


class DeclaredLimitStillRefusesTests(unittest.TestCase):
    """A declared limit admits the recording's own movement, not everything."""

    def test_a_spike_beyond_the_declared_limit_is_still_a_fault(self):
        # The declared limit is 40x the chain default, so the obvious worry is that
        # it admits anything. It does not: an electrode that jumps 100 000 uV from
        # the anchor is still named, and still counted in the census the session's
        # signal_quality verdict is read from.
        eeg, time = trace(drift_uv=100.0)
        column = CHANNELS.index("Cz")
        eeg[1500:1520, column] += SPIKE_UV
        processor, windows = fit(eeg, time, amplitude_limit_uv=DECLARED_UV)
        faults = processor.quality.fault_channels(time[0], time[-1])
        self.assertIn("amplitude", faults)
        self.assertIn("Cz", faults["amplitude"])
        self.assertIn("Cz", processor.quality.bad_channels(time[0], time[-1]))
        self.assertTrue(any(window.bad_channels for window in windows))

    def test_the_same_spike_is_a_fault_at_the_default_too(self):
        # The other half of the premise: this spike is not a fault *because* the
        # limit was declared, it is a fault at either value. The declaration is a
        # threshold, not a switch.
        eeg, time = trace(drift_uv=100.0)
        eeg[1500:1520, CHANNELS.index("Cz")] += SPIKE_UV
        processor, _ = fit(eeg, time)
        self.assertIn("Cz", processor.quality.fault_channels(
            time[0], time[-1]
        )["amplitude"])

    def test_the_declared_limit_cannot_separate_this_recording_s_own_movement(self):
        # The honest limit of the whole mechanism, asserted rather than only
        # described: a movement *inside* the declared limit is admitted on purpose,
        # and on a recording whose own movement is 20 mV that means the amplitude
        # criterion has no selectivity left - it can no longer tell "the electrode
        # drifted" from "the electrode popped". This is the finding the declaration
        # costs, and it is why the value belongs in the run record.
        eeg, time = trace(drift_uv=100.0)
        eeg[1500:1520, CHANNELS.index("Cz")] += DECLARED_UV * 0.5
        declared, _ = fit(eeg, time, amplitude_limit_uv=DECLARED_UV)
        default, _ = fit(eeg, time)
        band = (time[0], time[-1])
        self.assertEqual(declared.quality.fault_channels(*band), {})
        self.assertIn("Cz", default.quality.fault_channels(*band)["amplitude"])

    def test_a_declared_limit_below_the_recording_is_refused_nothing(self):
        # Declaring a limit under the recording's own movement reproduces the
        # measured failure exactly, which is why the declaration is measured
        # first: this is the state the ANT route was in at the default.
        eeg, time = trace(drift_uv=2500.0)
        processor, windows = fit(eeg, time, amplitude_limit_uv=1000.0)
        band = (time[0], time[-1])
        self.assertIn("amplitude", processor.quality.fault_channels(*band))
        self.assertTrue(all(window.bad_channels for window in windows))


class RunnerMeasuresBeforeItDeclaresTests(unittest.TestCase):
    """The declaration is accounted against a measurement of the same window."""

    def test_the_profile_counts_electrodes_on_both_sides_of_the_limit(self):
        eeg, _ = trace(drift_uv=2500.0)
        # An electrode that ramps to the amplifier's rail: it starts at the anchor,
        # so its excursion is the whole ramp - the case the declaration exists for,
        # and one that must still be counted at the declared limit.
        eeg[:, CHANNELS.index("F8")] = np.linspace(100.0, RAIL_UV, len(eeg))
        profile = ant_live.excursion_profile(
            eeg, CHANNELS, declared_limit_uv=DECLARED_UV
        )
        self.assertEqual(profile["declared_limit_uv"], DECLARED_UV)
        self.assertEqual(profile["default_limit_uv"], 500.0)
        self.assertIn("F8", profile["faulted_at_declared"])
        self.assertIn("F8", profile["faulted_at_default"])
        self.assertIn("Cz", profile["faulted_at_default"])
        self.assertNotIn("Cz", profile["faulted_at_declared"])
        self.assertLess(profile["max_excursion_on_unfaulted_microvolts"], DECLARED_UV)

    def test_an_electrode_sitting_at_the_rail_is_not_an_excursion(self):
        # The measurement must not confuse "railed" with "moved": an electrode
        # that is already at 83 333 uV on the anchor row never moves from it, so
        # it is the saturation fault's business, not the amplitude fault's. This
        # is the operator's own F8, and it is why this run still relies on
        # saturation and the channel census to name it.
        eeg, _ = trace(drift_uv=2500.0)
        eeg[:, CHANNELS.index("F8")] = RAIL_UV
        profile = ant_live.excursion_profile(
            eeg, CHANNELS, declared_limit_uv=DECLARED_UV
        )
        self.assertNotIn("F8", profile["faulted_at_declared"])
        self.assertEqual(
            profile["per_channel"]["F8"]["excursion_max_microvolts"], 0.0
        )

    def test_the_profile_anchors_where_the_monitor_anchors(self):
        # The monitor anchors at the first row it is fed; a profile that used the
        # mean would report a number the chain never uses.
        eeg, _ = trace(drift_uv=2500.0)
        profile = ant_live.excursion_profile(
            eeg, CHANNELS, declared_limit_uv=DECLARED_UV
        )
        expected = float(np.abs(eeg[:, 0] - eeg[0, 0]).max())
        self.assertAlmostEqual(
            profile["per_channel"][CHANNELS[0]]["excursion_max_microvolts"],
            expected, places=6,
        )

    def test_the_profile_refuses_a_trace_it_cannot_label(self):
        eeg, _ = trace(samples=100)
        with self.assertRaises(ValueError):
            ant_live.excursion_profile(eeg, CHANNELS[:-1], declared_limit_uv=DECLARED_UV)


class SessionWiringTests(unittest.TestCase):
    """The policy reaches the chain the session builds, not just the dataclass."""

    def source(self):
        return SimpleNamespace(
            channel_names=CHANNELS, sample_rate=RATE, reference="CPz",
            upstream_processing="unit test fixture", start=0.0,
        )

    def decoder(self):
        return SimpleNamespace(
            contract={
                "eeg_channels": list(CHANNELS), "window_seconds": 5.0,
                "step_seconds": 1.0, "input_sfreq": RATE, "bandpass": [0.5, 32.0],
                "input_reference": "CPz", "upstream_processing": "unit test fixture",
            },
            config=SimpleNamespace(sample_rate=64.0, band=(0.5, 32.0), lag_samples=10),
        )

    def session(self, **policy):
        values = {"check_channels": False, "max_bad_channels": 0}
        values.update(policy)
        return AttentionSession(
            source=self.source(), decoder=self.decoder(), references=None,
            policy=RunPolicy(**values),
        )

    def test_the_session_hands_the_declared_limit_to_the_chain(self):
        session = self.session(amplitude_limit_uv=DECLARED_UV)
        self.assertEqual(session.settings.amplitude_limit_uv, DECLARED_UV)
        processor = AuditoryProcessor(session.settings)
        self.assertEqual(processor.quality._amplitude_limit, DECLARED_UV)
        self.assertEqual(processor.repair._amplitude_uv, DECLARED_UV)

    def test_a_session_that_declares_nothing_keeps_the_chain_default(self):
        session = self.session()
        self.assertIsNone(session.settings.amplitude_limit_uv)
        processor = AuditoryProcessor(session.settings)
        self.assertEqual(processor.quality._amplitude_limit, 500.0)
        self.assertEqual(processor.repair._amplitude_uv, 500.0)


MUTATIONS = (
    {
        "name": "the-declared-value-never-reaches-the-monitor",
        "file": "src/nova2026/auditory/streaming.py",
        "before": '            exclude_channels=exclude_channels, **self._quality_limits(settings),',
        "after": "            exclude_channels=exclude_channels,",
        "test": "PointingAtOneOwnerTests.test_a_declared_limit_reaches_the_monitor_and_the_repairer",
    },
    {
        "name": "the-declared-value-never-reaches-the-repairer",
        "file": "src/nova2026/auditory/streaming.py",
        "before": '        if amplitude is not None:\n            limits["amplitude_limit_uv"] = float(amplitude)',
        "after": '        if False:\n            limits["amplitude_limit_uv"] = float(amplitude)',
        "test": "PointingAtOneOwnerTests.test_a_declared_limit_reaches_the_monitor_and_the_repairer",
    },
    {
        "name": "the-policy-field-is-never-handed-to-the-chain",
        "file": "src/nova2026/auditory/session.py",
        "before": "        settings.amplitude_limit_uv = policy.amplitude_limit_uv",
        "after": "        pass",
        "test": "SessionWiringTests.test_the_session_hands_the_declared_limit_to_the_chain",
    },
    {
        "name": "the-declared-limit-is-not-what-the-profile-judges-with",
        "file": "scripts/getlive/ant_live.py",
        "before": '            "faulted_at_limit": bool(peak[index] > declared_limit_uv),',
        "after": '            "faulted_at_limit": bool(peak[index] > default_limit_uv),',
        "test": "RunnerMeasuresBeforeItDeclaresTests.test_the_profile_counts_electrodes_on_both_sides_of_the_limit",
    },
    {
        "name": "the-declared-value-is-not-recorded-in-the-run-policy",
        "file": "src/nova2026/auditory/session.py",
        "before": '            "amplitude_limit_uv": self.amplitude_limit_uv,\n',
        "after": "            # the declaration is not recorded\n",
        "test": "PointingAtOneOwnerTests.test_the_run_policy_records_the_value_it_declared",
    },
)
"""The mutations this mechanism's tests must catch, and the test that catches each.

Each entry names the file, the exact text and the exact replacement that removes
or falsifies the behaviour. They are applied to a **copy** of the tree, never to
the working tree.
"""


def run_mutation_check(repo: Path, *, timeout: float = 900.0) -> dict:
    """Apply each mutation in a copied tree and report whether its test went red.

    The copy is what makes this safe to run inside the suite: the real tree is
    read and never written. Three traps had to be closed before its verdicts
    meant anything, and all three were measured here:

    1. **The editable install wins over the copy.** ``.venv`` carries
       ``__editable__.nova2026-0.1.0.pth``, which puts the *original*
       ``src`` on ``sys.path`` from ``site-packages``. A child that merely runs
       in the copy still imports the unmutated module and reports a false
       survivor, which is exactly what the first version of this check did (4 of
       5 mutations "survived"). The child therefore inserts its own tree at the
       front of ``sys.path`` before importing anything.
    2. **Stale bytecode.** A ``.pyc`` written from the pristine file can still
       match a mutated source whose mtime lands in the same second, so every
       ``__pycache__`` in the copy is removed before each run and the child runs
       with ``-B``.
    3. **A mutation that does not parse is not a mutation.** It is compiled
       before it is run and reported as invalid, because a syntax error also
       makes the test "fail" while proving nothing about the behaviour.
    """

    import shutil
    import tempfile

    report = {"kind": "mutation check for the declared amplitude limit", "results": []}
    with tempfile.TemporaryDirectory(prefix="nova-amplitude-mutation-") as scratch:
        copy = Path(scratch) / "repo"
        shutil.copytree(
            repo, copy,
            ignore=shutil.ignore_patterns(".git", ".venv", "node_modules", "__pycache__"),
        )
        for cache in copy.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        runner = (
            "import sys\n"
            "sys.path[:0] = ['', {root!r}, {src!r}]\n"
            "import unittest\n"
            "unittest.main(module=None, argv=['unittest', '-v', {test!r}])\n"
        )
        for mutation in MUTATIONS:
            target = copy / mutation["file"]
            original = target.read_text(encoding="utf-8")
            if original.count(mutation["before"]) != 1:
                report["results"].append({
                    "mutation": mutation["name"], "caught": False,
                    "detail": (
                        f"the mutation text appears {original.count(mutation['before'])} "
                        "times in the source, not once: the mutation itself is stale"
                    ),
                })
                continue
            mutated = original.replace(mutation["before"], mutation["after"], 1)
            target.write_text(mutated, encoding="utf-8")
            for cache in copy.rglob("__pycache__"):
                shutil.rmtree(cache, ignore_errors=True)
            try:
                compile(mutated, str(target), "exec")
            except SyntaxError as error:
                # A mutation that does not parse is not a mutation: its "red"
                # test would prove nothing, so it is reported as invalid rather
                # than counted as caught.
                target.write_text(original, encoding="utf-8")
                report["results"].append({
                    "mutation": mutation["name"],
                    "caught_by": mutation["test"],
                    "caught": False,
                    "invalid_mutation": True,
                    "detail": f"the mutated source does not parse: {error}",
                })
                continue
            result = subprocess.run(
                [sys.executable, "-B", "-c", runner.format(
                    root=str(copy), src=str(copy / "src"), test=mutation["test"]
                )],
                cwd=str(copy), capture_output=True, text=True, timeout=timeout,
            )
            target.write_text(original, encoding="utf-8")
            report["results"].append({
                "mutation": mutation["name"],
                "caught_by": mutation["test"],
                "caught": result.returncode != 0,
                "returncode": result.returncode,
                "detail": (result.stderr or result.stdout).strip().splitlines()[-1:],
            })
    report["survivors"] = [
        row["mutation"] for row in report["results"] if not row["caught"]
    ]
    report["kind"] = (
        "each mutation removes one piece of the declared-limit mechanism and is "
        "applied to a copy of the tree with its bytecode caches cleared; a mutation "
        "its named test does not catch is a test that is not evidence (plan "
        "section 6.4), and a mutation that does not parse is reported as invalid "
        "rather than as caught"
    )
    return report


class MeasuredRecordingTests(unittest.TestCase):
    """On the operator's real session, the declaration is what clears the census.

    The recording is read-only and may be absent (``tmp/`` is git-ignored), so this
    skips - loudly, with the reason - rather than pretending to have run. The
    numbers it asserts are the measured ones: 13 of the 20 electrodes move further
    than the chain's 500 uV default from the window's own anchor, F3 and F8 are
    railed and excluded, and the declared 4 000 uV admits every other electrode.
    """

    SESSION = REPO / "datasets" / "AAD-ANT" / "session_19-34-06.npz"

    def test_the_declared_limit_is_above_the_recording_s_own_measured_excursion(self):
        if not self.SESSION.is_file():
            self.skipTest(f"{self.SESSION} is not present; tmp/ is git-ignored")
        from scripts.getlive.ant_publish import load_ant_trial, slice_window

        trial = load_ant_trial(self.SESSION)
        window, _ = slice_window(trial, 5.0, 130.0)
        declared = ant_live.excursion_profile(
            window.eeg, tuple(trial.channel_names), declared_limit_uv=DECLARED_UV
        )
        # On the raw published window the chain's default faults 13 of the 20
        # electrodes: that is the measured reason no decision was ever emitted.
        self.assertEqual(len(declared["faulted_at_default"]), 13)
        self.assertIn("F3", declared["faulted_at_default"])
        self.assertIn("Cz", declared["faulted_at_default"])
        # It also shows why this window is the wrong point to measure: it is not
        # the stream the monitor is fed (the run record carries that measurement,
        # five times larger), so the declaration is checked against both.
        self.assertGreater(
            declared["max_excursion_on_unfaulted_microvolts"], 3000.0
        )
        self.assertLess(declared["max_excursion_microvolts"], DECLARED_UV)
        self.assertEqual(declared["faulted_at_declared"], [])
        excluded = ("F8", "F3")
        usable = [name for name in trial.channel_names if name not in excluded]
        self.assertEqual(len(usable), 18)
        # A railed electrode is not an excursion at all: F8 sits at 83 333.3 uV on
        # the anchor row and never moves from it, so this limit can never be what
        # names it. Saturation and the census are.
        self.assertEqual(
            declared["per_channel"]["F8"]["excursion_max_microvolts"], 0.0
        )
        self.assertFalse(declared["per_channel"]["F8"]["faulted_at_default"])

    def test_the_declared_limit_clears_every_electrode_at_the_chain_point(self):
        # The decisive measurement, on the stream the monitor is fed: after the
        # pre-chain resampler to the contract's rate, the declared limit must leave
        # only the two railed electrodes faulting. It is asserted here rather than
        # only recorded, because the first declared run measured at the wrong point
        # and left ten electrodes faulted while looking like a success.
        if not self.SESSION.is_file():
            self.skipTest(f"{self.SESSION} is not present; tmp/ is git-ignored")
        from scripts.getlive.ant_publish import load_ant_trial, slice_window

        model = RidgeDecoder.load(REPO / "models" / "auditory_kuleuven_live20.npz")
        trial = load_ant_trial(self.SESSION)
        window, _ = slice_window(trial, 5.0, 130.0)
        channels = tuple(model.contract["eeg_channels"])
        delivered = ant_live.chain_point_window(
            window, channels, float(model.contract["input_sfreq"])
        )
        self.assertIsNotNone(delivered, "the pre-chain resampler refused the window")
        data, _ = delivered
        self.assertAlmostEqual(data.shape[1], len(channels))
        profile = ant_live.excursion_profile(
            data, channels, declared_limit_uv=DECLARED_UV
        )
        self.assertEqual(sorted(profile["faulted_at_declared"]), ["F3", "F8"])
        self.assertEqual(len(profile["faulted_at_default"]), 20)
        # The margin over the largest admitted excursion is real but small, which
        # is the honest description of a limit derived from the recording.
        self.assertLess(
            profile["max_excursion_on_unfaulted_microvolts"], DECLARED_UV
        )
        self.assertGreater(
            profile["max_excursion_on_unfaulted_microvolts"], 0.9 * DECLARED_UV
        )


class MutationEvidenceTests(unittest.TestCase):
    """Each mutation of the mechanism must turn a named test red.

    A test that passes with the behaviour removed is not evidence (plan section
    6.4). This runs the mutations itself so the claim is reproducible instead of
    asserted: each one edits the source in a copy of the tree, clears the bytecode
    caches in that copy before running (a stale ``.pyc`` with a one-second mtime
    resolution can hide a mutation and manufacture a false survivor), runs one
    named test and reports whether it went red.
    """

    def test_every_mutation_is_caught_by_its_named_test(self):
        report = run_mutation_check(REPO)
        # Written before the assertions on purpose: when a mutation survives, the
        # report is the evidence of which one, and a report that only exists on
        # success would hide exactly the case it is for.
        out = REPO / "results" / "antneuro_amplitude_mutations.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        self.assertEqual(
            [row["mutation"] for row in report["results"] if not row["caught"]], []
        )
        self.assertEqual(len(report["results"]), len(MUTATIONS))


if __name__ == "__main__":  # pragma: no cover - manual runs
    unittest.main()
