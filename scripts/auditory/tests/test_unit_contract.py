"""The unit exponent stays parameterizable without moving any existing default.

Two datasets reach the same chain in two different units. The KU Leuven trials
are microvolts (peak absolute sample in the hundreds), and for microvolts
``Repair``'s endpoint check is an identity: ``10 ** (exponent + 6) == 1``. The
ANT rig records volts, whose correct exponent is 0. The chain used to hardcode
-6, which was right for the data it had only by accident and inexpressible for
the data it was about to get.

The first test is the important one: it pins the *historical* behaviour, so the
parameter cannot quietly change what every existing caller and trained decoder
was built against. The others pin the meaning of the new value and the channel
map step 5 must split on.
"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.streaming import (
    DEFAULT_SOURCE_UNIT_EXPONENT, SUPPORTED_SOURCE_UNIT_EXPONENTS, AuditoryProcessor,
    stream_config,
)

REPO = Path(__file__).resolve().parents[3]
METADATA = REPO / "datasets" / "AAD-KULeuven" / "metadata.json"


def source_trial(rate=128.0, channels=("F3", "F4")):
    """The source metadata ``stream_config`` reads, without converted data."""

    return SimpleNamespace(
        sample_rate=rate, channel_names=channels, reference="unknown",
        upstream_processing="synthetic",
    )


def to_uv(processor):
    """The endpoint-check factor, which no public reader exposes."""

    return vars(processor.repair)["_to_uv"]


class DefaultUnitTests(unittest.TestCase):
    """The default must be the behaviour every existing caller already had."""

    def test_default_settings_still_scale_endpoints_by_one(self):
        settings = stream_config(source_trial(), AuditoryConfig())
        self.assertEqual(settings.source_unit_exponent, -6)
        self.assertEqual(DEFAULT_SOURCE_UNIT_EXPONENT, -6)
        self.assertEqual(to_uv(AuditoryProcessor(settings)), 1.0)

    def test_settings_built_by_hand_keep_the_same_default(self):
        # Third-party settings objects (not built by stream_config) must not lose
        # the old behaviour just because the attribute is absent.
        settings = stream_config(source_trial(), AuditoryConfig())
        del settings.source_unit_exponent
        self.assertEqual(to_uv(AuditoryProcessor(settings)), 1.0)

    def test_signature_default_is_the_documented_constant(self):
        import inspect

        parameter = inspect.signature(stream_config).parameters["source_unit_exponent"]
        self.assertEqual(parameter.default, DEFAULT_SOURCE_UNIT_EXPONENT)


class VoltsUnitTests(unittest.TestCase):
    """A volts source is expressible, and the difference is visible."""

    def test_volts_source_scales_endpoints_by_a_million(self):
        settings = stream_config(source_trial(), AuditoryConfig(),
                                 source_unit_exponent=0)
        self.assertEqual(settings.source_unit_exponent, 0)
        self.assertEqual(to_uv(AuditoryProcessor(settings)), 1e6)

    def test_the_two_conventions_disagree_about_an_ordinary_microvolt_signal(self):
        # 325 uV is an ordinary KU Leuven sample. Under the volts convention the
        # same endpoints read as 3.25e8 uV, past Repair's 75000 uV saturation
        # line, so a repair between them is refused. This is the failure the
        # parameter makes expressible, shown as behaviour rather than as a claim.
        from nova2026.streaming import UnrepairableError
        from nova2026.streaming.preprocess import Repair

        times = np.arange(3) / 128.0
        damaged = np.array([[325.0, 0.0], [np.nan, 0.0], [325.0, 0.0]])
        microvolts = Repair(128.0, source_unit_exponent=-6)(damaged, times)[0]
        self.assertEqual(microvolts.shape, (3, 2))
        with self.assertRaises(UnrepairableError) as caught:
            Repair(128.0, source_unit_exponent=0)(damaged, times)
        self.assertEqual(caught.exception.kind, "unsafe_endpoints")

    def test_refuses_an_unsupported_exponent(self):
        with self.assertRaises(ValueError):
            stream_config(source_trial(), AuditoryConfig(), source_unit_exponent=-2)
        self.assertEqual(SUPPORTED_SOURCE_UNIT_EXPONENTS, (0, -3, -6, -9))

    def test_the_exponent_is_not_a_contract_key(self):
        # contract is compared against trained models, so a new key there would
        # invalidate every decoder; the exponent is run policy instead.
        common = stream_config(source_trial(), AuditoryConfig())
        volts = stream_config(source_trial(), AuditoryConfig(), source_unit_exponent=0)
        self.assertEqual(AuditoryProcessor(common).contract,
                         AuditoryProcessor(volts).contract)
        self.assertNotIn("source_unit_exponent", AuditoryProcessor(common).contract)


class ChannelMapTests(unittest.TestCase):
    """The 20 shared indices are the live contract, so pin them to the metadata."""

    def setUp(self):
        from scripts.auditory.kuleuven_contract import SHARED_CHANNELS

        self.shared = SHARED_CHANNELS

    def test_shared_indices_match_the_committed_channel_order(self):
        names = list(json.loads(METADATA.read_text(encoding="utf-8"))["channel_names"])
        self.assertEqual(len(names), 64)
        self.assertEqual([names[index] for _, index in self.shared],
                         [name for name, _ in self.shared])
        self.assertEqual(names[47], "Cz")

    def test_indices_are_unique_and_the_live_set_is_the_shared_plus_four(self):
        from scripts.auditory.kuleuven_contract import LIVE_CAP_CHANNELS, LIVE_ONLY_CHANNELS

        indices = [index for _, index in self.shared]
        self.assertEqual(len(set(indices)), 20)
        self.assertEqual(set(LIVE_CAP_CHANNELS),
                         {name for name, _ in self.shared} | set(LIVE_ONLY_CHANNELS))
        self.assertEqual(len(LIVE_CAP_CHANNELS), 24)


class GroupKeyTests(unittest.TestCase):
    """`rep_*` is the same story, which is what stops the split from leaking."""

    def test_repeat_folds_into_its_base(self):
        from scripts.auditory.kuleuven_contract import (
            canonical_group, canonical_story, group_key,
        )

        self.assertEqual(canonical_story("rep_part1_track1_dry.wav"),
                         "part1_track1_dry.wav")
        self.assertEqual(canonical_story("part1_track1_dry.wav"),
                         "part1_track1_dry.wav")
        base = ["part1_track1_dry.wav", "part1_track2_dry.wav"]
        repeat = ["rep_part1_track1_dry.wav", "rep_part1_track2_dry.wav"]
        self.assertEqual(group_key(base), group_key(repeat))
        self.assertEqual(canonical_group("|".join(repeat)), "|".join(base))

    def test_usable_audio_stops_at_the_last_eeg_sample(self):
        from scripts.auditory.kuleuven_contract import usable_audio_samples

        # 49792 samples at 128 Hz span 388.9921875 s. The last candidate sample
        # at or before that instant is index 17154555, so 17154556 samples stay
        # and 220844 of the 17375400 converted samples are dropped.
        usable = usable_audio_samples(49792, 128.0, 44100.0)
        self.assertEqual(usable, 17154556)
        self.assertLessEqual((usable - 1) / 44100.0, (49792 - 1) / 128.0)
        self.assertGreater(usable / 44100.0, (49792 - 1) / 128.0)
        self.assertAlmostEqual(17375400 - usable, 220844)


if __name__ == "__main__":
    unittest.main()
