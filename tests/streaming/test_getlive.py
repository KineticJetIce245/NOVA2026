"""getlive tests: cap profiles, outlet selection, channel policy, scoring.

Everything here is hardware-free and transport-free: the live script is tested
through its pure pieces (cap profiles, selection rules, acceptance rules,
reporting), and the one place that would need an amplifier - the stream loop -
is not exercised. Run from the repository root:

    .venv/Scripts/python.exe -B -m unittest tests.streaming.test_getlive -v
"""

import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.getlive import probe, report
from scripts.getlive.cap import (
    CAP_CHANNELS,
    CAP_EEG_CHANNELS,
    CAP_EOG_CHANNEL,
    CAP_GROUND,
    CAP_REFERENCE,
    CA208,
    MODEL_EEG_CHANNELS,
    MODEL_EXCLUDED_EEG_CHANNELS,
    CapProfile,
    declared_profile,
    resolve_channels,
    select_profile,
)
from scripts.getlive.checks import (
    FAIL,
    PASS,
    WARN,
    RunFacts,
    evaluate,
)
from scripts.getlive.electrodes import ChannelStats
from scripts.getlive.outlets import (
    filter_rows,
    find_outlets,
    format_outlets,
    is_eeg_like,
    outlet_rows,
    select_outlet,
)

# The legacy prototype that first encoded the CA-208 cap order. The new
# hardware contract must not drift away from it.
from scripts.dataproc.streaming.config import (
    CA208_CHANNELS,
    EEG_CHANNELS,
)


def row(
    name: str = "eego",
    stype: str = "EEG",
    channels: int = 64,
    sfreq: float = 500.0,
    source_id: str = "eego-1",
) -> dict:
    """One resolved-outlet row as ``outlets.outlet_rows`` produces it."""

    return {
        "name": name,
        "stype": stype,
        "source_id": source_id,
        "n_channels": channels,
        "sfreq": sfreq,
        "hostname": "host",
        "uid": f"uid-{name}",
        "labels_advertised": None,
    }


def facts(**overrides) -> RunFacts:
    """A clean, accepted run; tests override the fields they care about."""

    values = {
        "expected_sfreq": 500.0,
        "source_sfreq": 500.0,
        "expected_channels": 64,
        "source_channels": 64,
        "dropped_channels": (),
        "window_reasons": {},
        "warmup_windows": 4,
        "stats": {
            "blocks": 300,
            "samples": 15000,
            "windows": 52,
            "valid": 52,
            "rejected": 0,
            "gaps": 0,
            "max_gap": 0.0,
            "max_lag": 0.05,
            "recoveries": 0,
            "repairs": 0,
            "dropped": 0,
        },
        "resampler_quality": "LQ",
        "resampler_delay": 1.88,
        "out_sfreq": 128.0,
        "window_seconds": 2.0,
        "model_sfreq": 128.0,
        "model_window_seconds": 2.0,
        "flat_channels": (),
        "noisy_channels": (),
        "channel_count": 64,
        "flat_uv": 1.0,
        "noisy_uv": 200.0,
        "duration": 30.0,
        "elapsed": 30.0,
        "interrupted": False,
        "failure": None,
        "min_valid_windows": 3,
        "cap_profile": "CA-208",
        "cap_mode": "auto",
        "excluded_channels": (),
        "check_channels": True,
        "max_bad_channels": 0,
        "bad_channel_windows": {},
        "held_rows": 0,
        "model_channels": 57,
        "model_missing": (),
    }
    values.update(overrides)
    return RunFacts(**values)


def statuses(acceptance) -> dict:
    """Map check name to status for compact assertions."""

    return {check.name: check.status for check in acceptance.checks}


def details(acceptance) -> dict:
    """Map check name to its detail text."""

    return {check.name: check.detail for check in acceptance.checks}


class CapContractTests(unittest.TestCase):
    def test_cap_matches_the_datasheet_order(self) -> None:
        # Pinning order from the datasheet: connector 1 carries EOG as
        # channel 32, connector 2 continues to Oz.
        self.assertEqual(len(CAP_EEG_CHANNELS), 63)
        self.assertEqual(len(CAP_CHANNELS), 64)
        self.assertEqual(CAP_CHANNELS[0], "Fp1")
        self.assertEqual(CAP_CHANNELS[31], CAP_EOG_CHANNEL)
        self.assertEqual(CAP_CHANNELS[32], "AF7")
        self.assertEqual(CAP_CHANNELS[-1], "Oz")
        self.assertEqual(CAP_CHANNELS[15], "Cz")
        self.assertEqual(len(set(CAP_CHANNELS)), 64)
        self.assertEqual(CAP_REFERENCE, "CPz")
        self.assertEqual(CAP_GROUND, "AFz")
        # The reference and the ground are electrodes, not signal channels.
        self.assertNotIn("CPz", CAP_EEG_CHANNELS)
        self.assertNotIn("AFz", CAP_EEG_CHANNELS)

    def test_cap_does_not_drift_from_the_legacy_prototype(self) -> None:
        self.assertEqual(CAP_CHANNELS, CA208_CHANNELS)
        self.assertEqual(MODEL_EEG_CHANNELS, EEG_CHANNELS)

    def test_model_contract_is_the_cap_minus_the_excluded_electrodes(self) -> None:
        self.assertEqual(len(MODEL_EEG_CHANNELS), 57)
        for name in MODEL_EXCLUDED_EEG_CHANNELS:
            self.assertNotIn(name, MODEL_EEG_CHANNELS)

    def test_the_datasheet_profile_describes_the_cap(self) -> None:
        self.assertEqual(CA208.eeg_channels, CAP_EEG_CHANNELS)
        self.assertEqual(CA208.eog_channels, (CAP_EOG_CHANNEL,))
        self.assertEqual(CA208.reference, CAP_REFERENCE)
        self.assertEqual(CA208.ground, CAP_GROUND)
        # Contract order (EEG first, auxiliary last), not pin order: the
        # datasheet interleaves EOG as connector-1 pin 32, the chain does not.
        self.assertEqual(CA208.channels, CAP_EEG_CHANNELS + (CAP_EOG_CHANNEL,))
        self.assertEqual(set(CA208.channels), set(CAP_CHANNELS))
        self.assertEqual(CA208.n_eeg, 63)
        self.assertEqual(CA208.source, "datasheet")

    def test_profile_validation(self) -> None:
        with self.assertRaises(ValueError):
            CapProfile("empty", ())
        with self.assertRaises(ValueError):
            CapProfile("", ("F3",))
        with self.assertRaises(ValueError):
            CapProfile("dupe", ("F3", "F3"))
        with self.assertRaises(ValueError):
            CapProfile("overlap", ("F3",), ("F3",))
        with self.assertRaises(ValueError):
            CapProfile("blank", ("F3", ""))

    def test_resolve_channels_modes(self) -> None:
        eeg, eog = resolve_channels(CA208, "cap", "eog")
        self.assertEqual((len(eeg), eog), (63, ("EOG",)))
        eeg, eog = resolve_channels(CA208, "cap", "drop")
        self.assertEqual((len(eeg), eog), (63, ()))
        eeg, eog = resolve_channels(CA208, "model", "eog")
        self.assertEqual((eeg, eog), (MODEL_EEG_CHANNELS, ("EOG",)))
        with self.assertRaises(ValueError):
            resolve_channels(CA208, "everything")
        with self.assertRaises(ValueError):
            resolve_channels(CA208, "cap", "maybe")
        with self.assertRaises(TypeError):
            resolve_channels("CA-208")


class DeclaredProfileTests(unittest.TestCase):
    """A cap nobody wrote a table for is contracted from the outlet itself."""

    def test_declared_types_decide_the_split(self) -> None:
        profile = declared_profile(
            ("F3", "F4", "EOG", "Status"),
            ("eeg", "eeg", "eog", "misc"),
        )
        self.assertEqual(profile.eeg_channels, ("F3", "F4"))
        self.assertEqual(profile.eog_channels, ("EOG",))
        self.assertEqual(profile.other_auxiliary, ("Status",))
        self.assertEqual(profile.source, "outlet")
        self.assertEqual(profile.n_eeg, 2)

    def test_names_decide_when_no_type_is_declared(self) -> None:
        profile = declared_profile(("Fp1", "Cz", "EOG1", "TRIG", "ECG"))
        self.assertEqual(profile.eeg_channels, ("Fp1", "Cz"))
        self.assertEqual(profile.eog_channels, ("EOG1",))
        self.assertEqual(profile.other_auxiliary, ("TRIG", "ECG"))

    def test_an_auxiliary_name_beats_a_declared_eeg_type(self) -> None:
        # The eego software routinely labels the EOG drop lead as EEG.
        profile = declared_profile(("F3", "EOG"), ("eeg", "eeg"))
        self.assertEqual(profile.eeg_channels, ("F3",))
        self.assertEqual(profile.eog_channels, ("EOG",))

    def test_declared_validation(self) -> None:
        with self.assertRaises(ValueError):
            declared_profile(())
        with self.assertRaises(ValueError):
            declared_profile(("F3", "F3"))
        with self.assertRaises(ValueError):
            declared_profile(("F3",), ("eeg", "eog"))
        with self.assertRaises(ValueError):
            declared_profile(("EOG", "Status"))

    def test_operator_assertions_are_recorded(self) -> None:
        profile = declared_profile(("F3", "F4"), reference="Cz", ground="AFz")
        self.assertEqual(profile.reference, "Cz")
        self.assertEqual(profile.ground, "AFz")

    def test_model_subset_keeps_model_order_and_drops_unknowns(self) -> None:
        profile = declared_profile(("O1", "F3", "Cz", "X9"))
        # Cz is not part of the offline model contract; X9 is not an electrode.
        self.assertEqual(profile.model_subset(), ("F3", "O1"))
        # An explicit model list is honoured in the order it is given, and
        # filtered to the electrodes this cap actually has.
        self.assertEqual(profile.model_subset(("O1", "F3")), ("O1", "F3"))
        self.assertEqual(profile.model_subset(("Q9",)), ())

    def test_model_mode_needs_a_shared_electrode(self) -> None:
        profile = declared_profile(("X1", "X2"))
        with self.assertRaises(ValueError):
            resolve_channels(profile, "model", "eog")
        eeg, eog = resolve_channels(profile, "cap", "drop")
        self.assertEqual((eeg, eog), (("X1", "X2"), ()))

    def test_model_mode_intersects_for_another_cap(self) -> None:
        profile = declared_profile(("F3", "F4", "Cz", "O1"))
        eeg, _ = resolve_channels(profile, "model", "drop")
        self.assertEqual(eeg, ("F3", "F4", "O1"))


class ProfileSelectionTests(unittest.TestCase):
    def test_auto_uses_the_datasheet_when_it_fits(self) -> None:
        self.assertIs(select_profile("auto", CAP_CHANNELS), CA208)
        # Extra published channels do not stop the datasheet from fitting.
        self.assertIs(select_profile("auto", CAP_CHANNELS + ("Status",)), CA208)

    def test_auto_falls_back_to_the_declared_montage(self) -> None:
        profile = select_profile("auto", ("F3", "F4", "EOG"), ("eeg", "eeg", "eog"))
        self.assertEqual(profile.name, "declared")
        self.assertEqual(profile.eeg_channels, ("F3", "F4"))
        self.assertEqual(profile.eog_channels, ("EOG",))

    def test_ca_208_mode_demands_the_datasheet(self) -> None:
        with self.assertRaises(ValueError) as caught:
            select_profile("ca-208", ("F3", "F4", "EOG"), ("eeg", "eeg", "eog"))
        message = str(caught.exception)
        self.assertIn("does not publish the CA-208 contract", message)
        self.assertIn("--cap declared", message)

    def test_declared_mode_never_uses_the_datasheet(self) -> None:
        profile = select_profile("declared", CAP_CHANNELS)
        self.assertEqual(profile.name, "declared")
        self.assertEqual(profile.n_eeg, 63)
        self.assertEqual(profile.eog_channels, ("EOG",))

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_profile("everything", CAP_CHANNELS)


class OutletSelectionTests(unittest.TestCase):
    def test_outlet_rows_reads_advertised_labels(self) -> None:
        info = SimpleNamespace(
            name="eego",
            stype="EEG",
            source_id="abc",
            n_channels=64,
            sfreq=500.0,
            hostname="rig-pc",
            uid="uid-1",
            get_channel_names=lambda: ["Fp1", "Fp2"],
        )
        rows = outlet_rows([info])
        self.assertEqual(rows[0]["labels_advertised"], ("Fp1", "Fp2"))
        self.assertEqual(rows[0]["n_channels"], 64)

    def test_single_eeg_outlet_is_selected(self) -> None:
        rows = [row("eego", "EEG"), row("markers", "Markers", channels=0)]
        self.assertEqual(select_outlet(rows)["name"], "eego")

    def test_two_eeg_outlets_are_ambiguous(self) -> None:
        rows = [row("eego", "EEG"), row("eego-2", "EEG")]
        with self.assertRaises(RuntimeError) as caught:
            select_outlet(rows)
        self.assertIn("pass --stream-name", str(caught.exception))

    def test_channel_count_selects_when_no_eeg_type(self) -> None:
        rows = [row("amp", "eego", channels=64), row("markers", "Markers", channels=2)]
        self.assertEqual(
            select_outlet(rows, expected_channels=57)["name"], "amp"
        )

    def test_unknown_network_fails_loudly(self) -> None:
        with self.assertRaises(RuntimeError):
            select_outlet([])
        with self.assertRaises(RuntimeError) as caught:
            select_outlet([row("markers", "Markers", channels=2)])
        self.assertIn("No outlet looks like an EEG amplifier", str(caught.exception))

    def test_explicit_identity_is_required_to_be_unique(self) -> None:
        rows = [row("eego", "EEG"), row("other", "EEG")]
        self.assertEqual(select_outlet(rows, name="other")["name"], "other")
        with self.assertRaises(RuntimeError) as caught:
            select_outlet(rows, name="missing")
        self.assertIn("No outlet matches", str(caught.exception))
        two = [row("eego", "EEG"), row("eego", "EEG", source_id="other")]
        with self.assertRaises(RuntimeError) as caught:
            select_outlet(two, name="eego")
        self.assertIn("2 outlets match", str(caught.exception))

    def test_type_filter_is_case_insensitive(self) -> None:
        rows = [row("eego", "eeg")]
        self.assertEqual(filter_rows(rows, stream_type="EEG"), rows)
        self.assertEqual(select_outlet(rows, stream_type="EEG")["name"], "eego")

    def test_is_eeg_like_without_expected_channels(self) -> None:
        self.assertTrue(is_eeg_like(row("eego", "EEG")))
        self.assertFalse(is_eeg_like(row("markers", "Markers", channels=64)))
        self.assertTrue(
            is_eeg_like(row("markers", "Markers", channels=64), expected_channels=57)
        )
        self.assertFalse(
            is_eeg_like(row("markers", "Markers", channels=2), expected_channels=57)
        )

    def test_find_outlets_retries_until_an_outlet_appears(self) -> None:
        calls = []

        def resolver(timeout):
            calls.append(timeout)
            return [] if len(calls) == 1 else [SimpleNamespace(
                name="eego",
                stype="EEG",
                source_id="s",
                n_channels=64,
                sfreq=500.0,
                hostname="host",
                uid="uid",
                get_channel_names=lambda: None,
            )]

        rows = find_outlets(timeout=1.0, interval=0.001, resolver=resolver)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "eego")
        self.assertGreaterEqual(len(calls), 2)

    def test_find_outlets_gives_up_after_the_timeout(self) -> None:
        rows = find_outlets(timeout=0.05, interval=0.001, resolver=lambda _: [])
        self.assertEqual(rows, [])

    def test_format_outlets_handles_empty_and_rows(self) -> None:
        self.assertIn("No LSL outlet", format_outlets([]))
        text = format_outlets([row("eego", "EEG")])
        self.assertIn("eego", text)
        self.assertIn("channels", text)


class CapMatchTests(unittest.TestCase):
    def test_full_cap_in_order_matches(self) -> None:
        match = CA208.match(CAP_CHANNELS)
        self.assertEqual(match["profile"], "CA-208")
        self.assertEqual(match["eeg_present"], 63)
        self.assertEqual(match["eog_present"], 1)
        self.assertTrue(match["eeg_order"])
        self.assertEqual(match["eeg_missing"], ())
        self.assertEqual(match["eog_missing"], ())
        self.assertEqual(match["extra"], ())

    def test_missing_extra_and_reordered_channels_are_reported(self) -> None:
        names = tuple(reversed(CAP_CHANNELS[:5])) + ("Status",)
        match = CA208.match(names)
        self.assertFalse(match["eeg_order"])
        self.assertEqual(match["extra"], ("Status",))
        self.assertIn("Oz", match["eeg_missing"])
        self.assertEqual(match["eog_present"], 0)

    def test_model_subset_keeps_cap_order(self) -> None:
        match = CA208.match(MODEL_EEG_CHANNELS)
        self.assertEqual(match["eeg_present"], 57)
        self.assertTrue(match["eeg_order"])
        self.assertEqual(match["eog_present"], 0)

    def test_a_declared_profile_expects_exactly_what_it_was_built_from(self) -> None:
        profile = declared_profile(("F3", "F4", "EOG"), ("eeg", "eeg", "eog"))
        match = profile.match(("F3", "F4", "EOG", "Status", "TRIG"))
        self.assertEqual(match["source"], "outlet")
        self.assertEqual(match["eeg_expected"], 2)
        self.assertTrue(match["eeg_order"])
        self.assertEqual(match["extra"], ("Status", "TRIG"))


class ProbeFormattingTests(unittest.TestCase):
    def test_uniform_attribute_is_collapsed(self) -> None:
        self.assertEqual(
            probe.uniform_or_summary(("microvolts",) * 64),
            "microvolts (all 64 channels)",
        )
        mixed = probe.uniform_or_summary(("microvolts", "microvolts", "volts"))
        self.assertIn("microvolts x2", mixed)
        self.assertIn("volts x1", mixed)
        self.assertEqual(probe.uniform_or_summary(()), "not declared")

    def test_wrapped_names_stay_within_the_width(self) -> None:
        text = probe.wrap_names(CAP_CHANNELS, width=40)
        self.assertTrue(all(len(line) <= 44 for line in text.splitlines()))
        self.assertIn("Fp1", text)

    def test_cap_match_block_names_the_profile(self) -> None:
        text = probe.format_cap_match(CA208.match(CAP_CHANNELS))
        self.assertIn("CA-208 (datasheet)", text)
        self.assertIn("63/63", text)
        self.assertIn("1/1", text)
        self.assertIn("reference / ground", text)

    def test_cap_match_block_lists_what_is_missing(self) -> None:
        text = probe.format_cap_match(CA208.match(("F3", "Status")))
        self.assertIn("missing electrodes", text)
        self.assertIn("channels outside cap", text)
        self.assertIn("Status", text)

    def test_profile_block_reports_the_split(self) -> None:
        profile = declared_profile(("F3", "EOG", "Status"))
        text = probe.format_profile(profile)
        self.assertIn("declared (outlet)", text)
        self.assertIn("other aux", text)
        self.assertIn("Status", text)


class ChannelStatsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.channels = ("F3", "F4", "O1", "EOG")
        self.stats = ChannelStats(self.channels, flat_uv=1.0, noisy_uv=200.0)
        times = np.arange(256) / 128.0
        self.eeg = np.column_stack(
            [
                np.zeros(256),                     # dead electrode
                50.0 * np.sin(2 * np.pi * 10 * times),   # 100 uV peak-to-peak
                10.0 * np.sin(2 * np.pi * 10 * times),   # 20 uV peak-to-peak
            ]
        )
        self.eog = (1000.0 * np.sin(2 * np.pi * 3 * times)).reshape(-1, 1)

    def test_flat_and_noisy_electrodes_are_flagged(self) -> None:
        self.stats.update(self.eeg, self.eog)
        summary = self.stats.summary()
        self.assertEqual(summary["windows"], 1)
        self.assertEqual(summary["flat"], ("F3",))
        self.assertEqual(summary["noisy"], ("EOG",))
        self.assertAlmostEqual(summary["channels"][1]["mean_ptp_uv"], 100.0, places=6)

    def test_eog_can_be_absent(self) -> None:
        stats = ChannelStats(self.channels[:3], flat_uv=0.0)
        stats.update(self.eeg)
        self.assertEqual(stats.summary()["windows"], 1)
        self.assertEqual(stats.summary()["flat"], ())

    def test_columns_must_match_the_contract(self) -> None:
        with self.assertRaises(ValueError):
            self.stats.update(np.zeros((10, 2)))
        with self.assertRaises(ValueError):
            ChannelStats(())

    def test_non_finite_rows_do_not_distort_the_statistics(self) -> None:
        block = self.eeg.copy()
        block[0] = np.nan
        self.stats.update(block, self.eog)
        self.assertEqual(self.stats.summary()["windows"], 1)

    def test_worst_report_handles_an_empty_run(self) -> None:
        self.assertIn("no valid window", self.stats.format_worst())
        self.stats.update(self.eeg, self.eog)
        text = self.stats.format_worst(count=2)
        self.assertIn("flattest", text)
        self.assertIn("F3", text)


class AcceptanceTests(unittest.TestCase):
    def test_a_clean_run_is_accepted(self) -> None:
        acceptance = evaluate(facts())
        self.assertTrue(acceptance.ok)
        self.assertEqual(
            statuses(acceptance),
            {
                "connection": PASS,
                "source_rate": PASS,
                "channel_contract": PASS,
                "data_flow": PASS,
                "timing_gaps": PASS,
                "input_lag": PASS,
                "windows": PASS,
                "quality_reasons": PASS,
                "recovery": PASS,
                "repair": PASS,
                "resampler": PASS,
                "channel_scope": PASS,
                "bad_channels": PASS,
                "electrodes": PASS,
                "model_geometry": PASS,
                "model_channels": PASS,
                "duration": PASS,
            },
        )
        self.assertTrue(acceptance.to_dict()["ok"])
        self.assertIn("warm-up", details(acceptance)["quality_reasons"])
        self.assertIn("every EEG electrode is judged", details(acceptance)["channel_scope"])
        self.assertIn("no channel fault", details(acceptance)["bad_channels"])

    def test_a_failed_run_is_rejected(self) -> None:
        acceptance = evaluate(facts(failure="RuntimeError('boom')"))
        self.assertFalse(acceptance.ok)
        self.assertEqual(statuses(acceptance)["connection"], FAIL)

    def test_a_run_without_data_is_rejected(self) -> None:
        empty = {
            "samples": 0,
            "windows": 0,
            "valid": 0,
            "rejected": 0,
            "gaps": 0,
            "max_lag": 0.0,
            "recoveries": 0,
            "repairs": 0,
            "dropped": 0,
        }
        acceptance = evaluate(facts(stats=empty))
        self.assertFalse(acceptance.ok)
        self.assertEqual(statuses(acceptance)["data_flow"], FAIL)
        self.assertEqual(statuses(acceptance)["windows"], FAIL)

    def test_a_mismatched_rate_is_rejected(self) -> None:
        acceptance = evaluate(facts(source_sfreq=1000.0))
        self.assertFalse(acceptance.ok)
        self.assertEqual(statuses(acceptance)["source_rate"], FAIL)

    def test_a_fully_flat_cap_is_rejected(self) -> None:
        acceptance = evaluate(
            facts(flat_channels=CAP_CHANNELS, channel_count=len(CAP_CHANNELS))
        )
        self.assertFalse(acceptance.ok)
        self.assertEqual(statuses(acceptance)["electrodes"], FAIL)

    def test_cap_quality_problems_only_warn(self) -> None:
        acceptance = evaluate(
            facts(flat_channels=("F3", "F4"), noisy_channels=("O1",))
        )
        self.assertTrue(acceptance.ok)
        self.assertEqual(statuses(acceptance)["electrodes"], WARN)

    def test_transport_and_chain_events_only_warn(self) -> None:
        stats = dict(facts().stats)
        stats.update({"gaps": 2, "max_gap": 0.02, "max_lag": 0.8, "recoveries": 1,
                      "repairs": 5})
        acceptance = evaluate(
            facts(
                stats=stats,
                window_reasons={"amplitude": 3, "interpolated": 1},
                resampler_delay=2.5,
                dropped_channels=("EOG",),
                interrupted=True,
            )
        )
        self.assertTrue(acceptance.ok)
        self.assertEqual(statuses(acceptance)["timing_gaps"], WARN)
        self.assertEqual(statuses(acceptance)["input_lag"], WARN)
        self.assertEqual(statuses(acceptance)["recovery"], WARN)
        self.assertEqual(statuses(acceptance)["repair"], WARN)
        self.assertEqual(statuses(acceptance)["quality_reasons"], WARN)
        self.assertEqual(statuses(acceptance)["resampler"], WARN)
        self.assertEqual(statuses(acceptance)["channel_contract"], WARN)
        self.assertEqual(statuses(acceptance)["duration"], WARN)
        self.assertEqual(acceptance.summary()[FAIL], 0)

    def test_a_hopeless_resampler_delay_is_rejected(self) -> None:
        # HQ at 500 -> 128 Hz measured near 7.4 s in the package documentation:
        # by then the run cannot deliver timely windows at all.
        acceptance = evaluate(
            facts(resampler_quality="HQ", resampler_delay=7.4)
        )
        self.assertFalse(acceptance.ok)
        self.assertEqual(statuses(acceptance)["resampler"], FAIL)

    def test_a_run_at_the_acquire_guard_is_rejected(self) -> None:
        # The package stops the run when a consumed block is 3 s old.
        stats = dict(facts().stats)
        stats["max_lag"] = 3.5
        acceptance = evaluate(facts(stats=stats))
        self.assertFalse(acceptance.ok)
        self.assertEqual(statuses(acceptance)["input_lag"], FAIL)

    def test_a_slow_but_working_source_only_warns(self) -> None:
        stats = dict(facts().stats)
        stats["max_lag"] = 1.5
        acceptance = evaluate(facts(stats=stats))
        self.assertTrue(acceptance.ok)
        self.assertEqual(statuses(acceptance)["input_lag"], WARN)

    def test_offline_geometry_mismatch_only_warns(self) -> None:
        acceptance = evaluate(facts(out_sfreq=64.0))
        self.assertTrue(acceptance.ok)
        self.assertEqual(statuses(acceptance)["model_geometry"], WARN)

    def test_report_formats_every_check(self) -> None:
        text = evaluate(facts()).format()
        self.assertIn("USABLE", text)
        self.assertIn("electrodes", text)
        self.assertIn("bad_channels", text)


class ChannelPolicyTests(unittest.TestCase):
    """The channel-quality policy is reported, and the evidence is scored."""

    def test_exclusions_and_tolerance_are_reported(self) -> None:
        acceptance = evaluate(
            facts(excluded_channels=("Fp1",), max_bad_channels=2)
        )
        self.assertTrue(acceptance.ok)
        self.assertEqual(statuses(acceptance)["channel_scope"], PASS)
        detail = details(acceptance)["channel_scope"]
        self.assertIn("excluded from the verdict: Fp1", detail)
        self.assertIn("still recorded", detail)
        self.assertIn("up to 2 faulting channel(s) tolerated", detail)

    def test_turning_the_guard_off_warns(self) -> None:
        acceptance = evaluate(facts(check_channels=False))
        self.assertTrue(acceptance.ok)
        self.assertEqual(statuses(acceptance)["channel_scope"], WARN)
        self.assertIn("--no-channel-check", details(acceptance)["channel_scope"])

        with_exclusions = evaluate(
            facts(check_channels=False, excluded_channels=("Fp1", "F7"))
        )
        self.assertEqual(statuses(with_exclusions)["channel_scope"], WARN)
        self.assertIn("Fp1, F7", details(with_exclusions)["channel_scope"])

    def test_faulting_electrodes_are_named_with_their_counts(self) -> None:
        acceptance = evaluate(
            facts(
                excluded_channels=("Fp1",),
                bad_channel_windows={"Fp1": 12, "F7": 3},
                held_rows=40,
            )
        )
        self.assertEqual(statuses(acceptance)["bad_channels"], WARN)
        detail = details(acceptance)["bad_channels"]
        self.assertIn("Fp1 x12 window(s)", detail)
        self.assertIn("F7 x3 window(s)", detail)
        self.assertIn("40 row(s) held", detail)
        # A dead electrode is a contact problem, not a failed run.
        self.assertTrue(acceptance.ok)

    def test_model_channel_coverage_is_reported(self) -> None:
        acceptance = evaluate(facts(model_missing=("Fpz", "M1", "Cz")))
        self.assertTrue(acceptance.ok)
        self.assertEqual(statuses(acceptance)["model_channels"], WARN)
        detail = details(acceptance)["model_channels"]
        self.assertIn("54/57", detail)
        self.assertIn("Fpz, M1, Cz", detail)


class ParserTests(unittest.TestCase):
    def test_source_facts_must_be_stated(self) -> None:
        from scripts.getlive.live import main

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(["--duration", "1"])
        self.assertEqual(caught.exception.code, 2)
        message = stderr.getvalue()
        self.assertIn("--sfreq", message)
        self.assertIn("--source-units", message)
        self.assertIn("scripts.getlive.probe", message)

    def test_parser_defaults_suit_a_hardware_test(self) -> None:
        from scripts.getlive.live import build_parser

        args = build_parser().parse_args(
            ["--sfreq", "500", "--source-units", "uV"]
        )
        self.assertEqual(args.duration, 30.0)
        self.assertEqual(args.cap, "auto")
        self.assertEqual(args.channels, "cap")
        self.assertEqual(args.eog, "eog")
        self.assertEqual(args.out_sfreq, 128.0)
        self.assertEqual(args.window, 2.0)
        self.assertEqual(args.hop, 0.5)
        self.assertIsNone(args.record)
        self.assertIsNone(args.out)
        # Channel policy: the historical verdict unless the operator says so.
        self.assertFalse(args.no_channel_check)
        self.assertEqual(args.max_bad_channels, 0)
        self.assertEqual(args.exclude_channels, "")
        self.assertIsNone(args.reference)
        self.assertIsNone(args.ground)
        self.assertIsNone(args.expect_channels)

    def test_cap_and_channel_choices_are_validated(self) -> None:
        from scripts.getlive.live import build_parser

        base = ["--sfreq", "500", "--source-units", "uV"]
        for flag, value in (("--cap", "everything"), ("--channels", "all"),
                            ("--eog", "maybe")):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(base + [flag, value])

    def test_channel_list_parsing(self) -> None:
        from scripts.getlive.live import abbreviate, channel_list

        self.assertEqual(channel_list("Fp1, F7 ,, F8"), ("Fp1", "F7", "F8"))
        self.assertEqual(channel_list(""), ())
        with self.assertRaises(TypeError):
            channel_list(None)
        self.assertEqual(abbreviate(("F3", "F4")), "F3, F4")
        self.assertIn("+2 more", abbreviate(tuple("ABCDEF"), limit=4))


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        samples = 8
        self.window = SimpleNamespace(
            data=np.column_stack([np.zeros(samples), np.full(samples, 2.0)]),
            eog=np.zeros((samples, 0)),
            valid=True,
            reasons=(),
            segment=0,
            timestamps=np.arange(samples) / 128.0,
            bad_channels=(),
        )
        self.resampler = SimpleNamespace(
            out_sfreq=128.0,
            quality="LQ",
            startup_delay_seconds=1.88,
            output_samples=1744,
        )
        self.session = SimpleNamespace(
            window_samples=256,
            hop_samples=64,
            capacity_samples=768,
            out_sfreq=128.0,
            warmup_samples=256,
        )
        self.stats = SimpleNamespace(
            blocks=152,
            samples=7600,
            max_lag=1.042,
            gaps=0,
            windows=24,
            valid=20,
            rejected=4,
            recoveries=0,
            repairs=0,
            dropped=0,
        )
        self.args = SimpleNamespace(
            source_units="uV",
            notch=60.0,
            lpass=1.0,
            hpass=45.0,
            sfreq=500.0,
            duration=30.0,
            block=50,
            record=None,
        )

    def test_a_valid_window_line_carries_amplitude_and_lag(self) -> None:
        line = report.format_window(4, 2.5, 0.25, self.window)
        self.assertIn("win   4", line)
        self.assertIn("ptp_max=", line)
        self.assertIn("lag= 0.250s", line)
        self.assertNotIn("REJECT", line)
        self.assertNotIn("bad=", line)

    def test_an_invalid_window_line_names_its_reasons(self) -> None:
        window = SimpleNamespace(**vars(self.window))
        window.valid = False
        window.reasons = ("amplitude", "flatline")
        line = report.format_window(5, 3.0, 0.5, window)
        self.assertIn("REJECT", line)
        self.assertIn("amplitude,flatline", line)

    def test_a_tolerated_dead_electrode_stays_visible(self) -> None:
        window = SimpleNamespace(**vars(self.window))
        window.bad_channels = ("Fp1", "F7")
        line = report.format_window(6, 4.0, 0.2, window)
        self.assertIn("valid", line)
        self.assertIn("bad=Fp1,F7", line)

    def test_a_warmup_window_is_labelled_as_such(self) -> None:
        window = SimpleNamespace(**vars(self.window))
        window.valid = False
        window.reasons = ()
        self.assertIn("reasons=warmup", report.format_window(1, 0.5, 0.1, window))

    def test_setup_and_run_summaries_name_the_geometry(self) -> None:
        setup = report.format_setup(self.args, self.session, self.resampler)
        self.assertIn("resample 500->128 Hz", setup)
        self.assertIn("window=256", setup)
        self.assertNotIn("recording under", setup)
        run = report.format_run(self.stats, 14.0, self.resampler)
        self.assertIn("max_lag 1.042s", run)
        self.assertIn("20 valid", run)

    def test_the_channel_summary_names_the_profile_and_the_dead_list(self) -> None:
        text = report.format_channels(CA208, ("F3", "F4"), ("EOG",), ("F4",))
        self.assertIn("CA-208 (datasheet)", text)
        self.assertIn("2 EEG + 1 EOG (EOG)", text)
        self.assertIn("known dead (never fatal): F4", text)
        self.assertIn("reference CPz", text)

        declared = report.format_channels(
            declared_profile(("F3", "F4")), ("F3", "F4"), ()
        )
        self.assertIn("not asserted by the profile", declared)
        self.assertIn("no auxiliary", declared)

    def test_the_json_report_is_serializable_and_written(self) -> None:
        acceptance = evaluate(facts())
        stats = SimpleNamespace(to_dict=lambda: {"samples": 7600})
        recording = Path("runs/s1/a/run01/run01.sqlite")
        report_dict = report.build_report(
            args=self.args,
            source={"name": "eego", "channels": CAP_CHANNELS},
            acceptance=acceptance,
            stats=stats,
            electrodes={"windows": 20, "flat": (), "noisy": ()},
            resampler=self.resampler,
            recording=recording,
        )
        self.assertTrue(report_dict["accepted"])
        self.assertEqual(report_dict["recording"], str(recording))
        # Channel facts are optional: a caller that has none still gets JSON.
        self.assertNotIn("channels", report_dict)
        json.dumps(report_dict)  # must not raise: the report is JSON by contract

        destination = Path(".tmp_tests") / "getlive_report_test.json"
        report.write_report(destination, report_dict)
        self.assertTrue(destination.exists())
        written = json.loads(destination.read_text(encoding="utf-8"))
        self.assertTrue(written["accepted"])
        self.assertEqual(written["acceptance"]["counts"][FAIL], 0)

    def test_the_json_report_carries_the_channel_evidence(self) -> None:
        acceptance = evaluate(facts(bad_channel_windows={"Fp1": 4}))
        report_dict = report.build_report(
            args=self.args,
            source={"name": "eego", "channels": CAP_CHANNELS},
            acceptance=acceptance,
            stats=SimpleNamespace(to_dict=lambda: {"samples": 1}),
            electrodes={"windows": 20, "flat": (), "noisy": ()},
            resampler=self.resampler,
            recording=None,
            channels={
                "profile": "CA-208",
                "excluded_channels": ["Fp1"],
                "bad_channel_windows": {"Fp1": 4},
                "held_rows": 12,
            },
        )
        self.assertEqual(report_dict["channels"]["profile"], "CA-208")
        self.assertEqual(report_dict["channels"]["bad_channel_windows"], {"Fp1": 4})
        json.dumps(report_dict)


if __name__ == "__main__":
    unittest.main()
