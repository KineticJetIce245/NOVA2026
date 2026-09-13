"""The precomputed-envelope contract, on fixtures small enough to run anywhere.

Three things are pinned here, and all three are load-bearing for the demo:

1. The offline path and the runtime path produce the same numbers. A session
   decodes against a stored envelope while the live path would have computed one
   from the same audio, so any difference is a silent feature change.
2. Feeding the whole file equals feeding it in chunks. This is the property the
   plan requires before training, and it is asserted on synthetic speech plus a
   worst-case chunk length rather than on the dataset.
3. A stale envelope is refused. A source whose SHA256 no longer matches, or a
   file whose recorded parameters differ from the live ``AuditoryConfig``, must
   stop the session instead of being decoded with the wrong feature.
"""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from nova2026.auditory.config import AuditoryConfig
from nova2026.auditory.envelopes import (
    EnvelopeExtractor,
    EnvelopeVerificationError,
    envelope_timestamps,
    extract_envelope,
    load_envelope,
    verify_envelope_metadata,
)
from scripts.auditory import envelopes as cli

RATE = 8000
SECONDS = 6


def synthetic_speech(rate=RATE, seconds=SECONDS):
    """A deterministic broadband signal with a speech-like 4 Hz modulation."""

    times = np.arange(int(rate * seconds)) / rate
    carrier = np.sin(2 * np.pi * 320 * times) + 0.6 * np.sin(2 * np.pi * 1500 * times)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * times)
    return (0.4 * carrier * modulation).astype(float)


def write_wav(path, samples, rate=RATE):
    """Write float samples as 16-bit PCM, the format every stimulus uses."""

    wavfile.write(path, rate, np.round(samples * 32767).astype(np.int16))
    return Path(path)


class OfflineMatchesRuntimeTests(unittest.TestCase):
    """The offline whole-file entry point is the runtime extractor, not a twin."""

    def test_mono_is_fed_as_two_identical_columns_and_column_zero_is_kept(self):
        samples = synthetic_speech()
        config = AuditoryConfig()
        offline = extract_envelope(samples, RATE, config)
        # The runtime call site (TimestampedAudio, runner) passes both candidates.
        runtime = EnvelopeExtractor(RATE, config).feed(
            np.column_stack([samples, samples])
        )
        self.assertEqual(offline.shape, (len(runtime), 1))
        self.assertEqual(offline.dtype, np.float32)
        # Exact, not approximate: the two paths are the same code on the same
        # input, and an allclose tolerance would hide a re-derived grid.
        np.testing.assert_array_equal(offline[:, 0], runtime[:, 0].astype(np.float32))

    def test_explicit_two_column_input_uses_column_zero(self):
        left = synthetic_speech()
        right = np.roll(left, 13)
        config = AuditoryConfig()
        offline = extract_envelope(np.column_stack([left, right]), RATE, config)
        runtime = EnvelopeExtractor(RATE, config).feed(np.column_stack([left, right]))
        np.testing.assert_array_equal(offline[:, 0], runtime[:, 0].astype(np.float32))
        self.assertFalse(np.array_equal(offline[:, 0], runtime[:, 1].astype(np.float32)))

    def test_whole_file_equals_chunked_feeding(self):
        samples = synthetic_speech()
        config = AuditoryConfig()
        whole = extract_envelope(samples, RATE, config)
        extractor = EnvelopeExtractor(RATE, config)
        parts = [
            extractor.feed(np.column_stack([samples[start : start + 257]] * 2))
            for start in range(0, len(samples), 257)
        ]
        chunked = np.concatenate(parts)[:, :1]
        self.assertEqual(whole.shape, chunked.shape)
        maximum = float(np.max(np.abs(whole - chunked.astype(np.float32))))
        self.assertEqual(maximum, 0.0)

    def test_length_and_causal_first_sample_follow_the_runtime_grid(self):
        samples = synthetic_speech()
        config = AuditoryConfig()
        envelope = extract_envelope(samples, RATE, config)
        expected = int(np.floor((len(samples) - 1) * config.sample_rate / RATE)) + 1
        self.assertEqual(len(envelope), expected)
        timestamps = envelope_timestamps(len(envelope), config.sample_rate)
        self.assertEqual(timestamps[0], 0.0)
        np.testing.assert_allclose(np.diff(timestamps), 1 / config.sample_rate)

    def test_rejects_an_input_it_cannot_interpret(self):
        config = AuditoryConfig()
        with self.assertRaises(ValueError):
            extract_envelope(np.zeros((10, 3)), RATE, config)
        with self.assertRaises(ValueError):
            extract_envelope(np.full((10, 2), np.nan), RATE, config)
        with self.assertRaises(ValueError):
            extract_envelope(np.zeros(10), 0, config)


class RoundTripTests(unittest.TestCase):
    """What the CLI writes is what the verifier reads back, provenance included."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = AuditoryConfig()
        self.audio = write_wav(self.root / "candidate_a.wav", synthetic_speech())
        envelope, timestamps, metadata = cli.convert_audio(self.audio, self.config)
        self.path = self.root / "candidate_a.npz"
        cli.save_envelope(self.path, envelope, timestamps, metadata)

    def tearDown(self):
        self.directory.cleanup()

    def test_npz_holds_envelope_timestamps_and_json_metadata(self):
        with np.load(self.path, allow_pickle=False) as stored:
            self.assertEqual(set(stored.files), {"envelope", "timestamps", "metadata"})
            self.assertEqual(stored["envelope"].dtype, np.float32)
            self.assertEqual(stored["envelope"].ndim, 2)
            self.assertEqual(stored["envelope"].shape[1], 1)
            self.assertEqual(stored["timestamps"].dtype, np.float64)
            self.assertIsInstance(str(stored["metadata"]), str)
        self.assertNotIn(b"pickle", self.path.read_bytes())

    def test_metadata_carries_the_recorded_provenance(self):
        record = load_envelope(self.path, self.config)
        metadata = record.metadata
        for key in (
            "source_path",
            "source_sha256",
            "source_rate",
            "source_samples",
            "audio_rate",
            "sample_rate",
            "band",
            "envelope_method",
            "generated_at",
            "generator_version",
        ):
            self.assertIn(key, metadata)
        recorded = self.config.to_dict()
        self.assertEqual(metadata["sample_rate"], recorded["sample_rate"])
        self.assertEqual(metadata["band"], list(recorded["band"]))
        self.assertEqual(metadata["envelope_method"], recorded["envelope_method"])
        self.assertEqual(metadata["audio_rate"], RATE)
        self.assertEqual(metadata["source_samples"], len(synthetic_speech()))
        self.assertEqual(
            metadata["source_sha256"],
            hashlib.sha256(self.audio.read_bytes()).hexdigest(),
        )
        self.assertEqual(Path(metadata["source_path"]), self.audio.resolve())
        self.assertEqual(len(record), len(record.timestamps))
        np.testing.assert_allclose(record.timestamps[1] - record.timestamps[0], 0.015625)

    def test_a_missing_envelope_is_refused_rather_than_computed(self):
        with self.assertRaises(EnvelopeVerificationError):
            load_envelope(self.root / "absent.npz", self.config)

    def test_mismatched_sample_count_is_refused(self):
        with np.load(self.path, allow_pickle=False) as stored:
            envelope = stored["envelope"]
            timestamps = stored["timestamps"]
            metadata = json.loads(str(stored["metadata"]))
        cli.save_envelope(
            self.path, envelope[:-3], timestamps, metadata
        )
        with self.assertRaises(EnvelopeVerificationError):
            load_envelope(self.path, self.config)

    def test_a_wrong_band_in_the_file_is_refused(self):
        # A band recorded as something the session would never ask for: the
        # envelope's numbers cannot be reinterpreted, so it must not load.
        self.rewrite(band=[2.0, 11.0])
        with self.assertRaises(EnvelopeVerificationError):
            load_envelope(self.path, self.config)

    def test_a_different_band_in_the_session_is_refused(self):
        with self.assertRaises(EnvelopeVerificationError):
            load_envelope(self.path, AuditoryConfig(band=(2.0, 11.0)))

    def test_a_different_sample_rate_in_the_session_is_refused(self):
        with self.assertRaises(EnvelopeVerificationError):
            load_envelope(self.path, AuditoryConfig(sample_rate=128))

    def test_a_different_envelope_method_is_refused(self):
        self.rewrite(envelope_method="mean-abs-v0")
        with self.assertRaises(EnvelopeVerificationError):
            load_envelope(self.path, self.config)

    def test_changed_source_audio_is_refused(self):
        # Same name, different samples: the classic stale-envelope trap.
        write_wav(self.audio, synthetic_speech() * 0.5)
        with self.assertRaises(EnvelopeVerificationError) as caught:
            load_envelope(self.path, self.config, audio_path=self.audio)
        self.assertIn("SHA256", str(caught.exception))

    def test_a_relocated_tree_still_verifies_through_the_recorded_relative_path(self):
        # A clone or a copied tree loses the generating machine's absolute paths.
        # The recorded path is retried relative to the envelope's grandparent, so
        # the copy is still checked against its own audio instead of being trusted.
        moved = self.root / "elsewhere" / "audio"
        moved.mkdir(parents=True)
        relocated_audio = self.root / "elsewhere" / "candidate_a.wav"
        relocated_audio.write_bytes(self.audio.read_bytes())
        relocated = moved / self.path.name
        relocated.write_bytes(self.path.read_bytes())
        self.audio.unlink()
        record = load_envelope(relocated, self.config)
        self.assertEqual(record.metadata["source_sha256"], hashlib.sha256(relocated_audio.read_bytes()).hexdigest())

    def test_metadata_helper_rejects_a_missing_key(self):
        with np.load(self.path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"]))
        del metadata["source_sha256"]
        with self.assertRaises(EnvelopeVerificationError):
            verify_envelope_metadata(metadata, self.config)

    def test_metadata_helper_rejects_a_non_numeric_source_rate(self):
        with np.load(self.path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"]))
        metadata["source_rate"] = "44100"
        with self.assertRaises(EnvelopeVerificationError):
            verify_envelope_metadata(metadata, self.config)

    def test_metadata_helper_accepts_the_recorded_parameters(self):
        with np.load(self.path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"]))
        self.assertEqual(
            verify_envelope_metadata(metadata, self.config), metadata
        )

    def rewrite(self, **changes):
        """Rewrite the stored metadata with fields replaced, for defect tests."""

        with np.load(self.path, allow_pickle=False) as stored:
            envelope = stored["envelope"]
            timestamps = stored["timestamps"]
            metadata = json.loads(str(stored["metadata"]))
        metadata.update(changes)
        cli.save_envelope(self.path, envelope, timestamps, metadata)


class CommandLineTests(unittest.TestCase):
    """The pre-registered command, minus the dataset."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.audio = write_wav(self.root / "candidate_a.wav", synthetic_speech())
        self.out = self.root / "audio"

    def tearDown(self):
        self.directory.cleanup()

    def run_cli(self, *argv):
        # The CLI reports to stdout; the tests assert on files and exit codes.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            return cli.main([str(item) for item in argv])

    def test_generates_one_npz_per_input_and_refuses_to_replace_it(self):
        self.assertEqual(self.run_cli("--audio", self.audio, "--out", self.out), 0)
        produced = self.out / "candidate_a.npz"
        self.assertTrue(produced.is_file())
        self.assertEqual(list(self.out.glob("*.npz")), [produced])
        # Second run without --force is refused, and argparse exits non-zero.
        with self.assertRaises(SystemExit) as caught:
            self.run_cli("--audio", self.audio, "--out", self.out)
        self.assertNotEqual(caught.exception.code, 0)
        self.assertEqual(self.run_cli("--audio", self.audio, "--out", self.out, "--force"), 0)

    def test_audio_dir_selects_the_pattern_and_verify_rechecks_the_hashes(self):
        self.run_cli("--audio-dir", self.root, "--pattern", "*.wav", "--out", self.out)
        self.assertTrue((self.out / "candidate_a.npz").is_file())
        self.assertEqual(self.run_cli("--out", self.out, "--verify"), 0)
        write_wav(self.audio, synthetic_speech() * 0.25)
        self.assertEqual(self.run_cli("--out", self.out, "--verify"), 1)

    def test_two_inputs_sharing_a_stem_are_refused(self):
        other = self.root / "nested"
        other.mkdir()
        write_wav(other / "candidate_a.wav", synthetic_speech())
        with self.assertRaises(SystemExit):
            self.run_cli("--audio", self.audio, other / "candidate_a.wav", "--out", self.out)

    def test_more_than_two_channels_are_refused(self):
        wide = self.root / "wide.wav"
        wavfile.write(wide, RATE, np.zeros((RATE, 6), dtype=np.int16))
        with self.assertRaises(SystemExit) as caught:
            self.run_cli("--audio", wide, "--out", self.out)
        self.assertNotEqual(caught.exception.code, 0)
        self.assertEqual(list(self.out.glob("*.npz")) if self.out.is_dir() else [], [])

    def test_stereo_is_downmixed_the_way_the_decoder_reads_it(self):
        stereo = self.root / "stereo.wav"
        samples = synthetic_speech()
        wavfile.write(
            stereo, RATE, np.round(np.column_stack([samples, samples]) * 32767).astype(np.int16)
        )
        self.assertEqual(self.run_cli("--audio", stereo, "--out", self.out), 0)
        record = load_envelope(self.out / "stereo.npz", AuditoryConfig())
        mono = extract_envelope(samples, RATE, AuditoryConfig())
        np.testing.assert_allclose(record.envelope[:, 0], mono[:, 0], atol=2e-4)


class PackageTests(unittest.TestCase):
    """The CLI is importable as a module, which is how the plan invokes it."""

    def test_cli_module_imports_with_the_documented_entry_point(self):
        from scripts.auditory import envelopes

        self.assertTrue(callable(envelopes.main))
        self.assertEqual(Path(envelopes.__file__).name, "envelopes.py")


if __name__ == "__main__":
    unittest.main()
