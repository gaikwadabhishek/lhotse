import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
import torch.testing

from lhotse import CutSet, Fbank, MonoCut
from lhotse.dataset import AudioSamples, OnTheFlyFeatures, PrecomputedFeatures


@pytest.fixture
def libri_cut_set():
    cuts = CutSet.from_json("test/fixtures/libri/cuts.json")
    return CutSet.from_cuts(
        [
            cuts[0],
            cuts[0].with_id("copy-1"),
            cuts[0].with_id("copy-2"),
            cuts[0].append(cuts[0]),
        ]
    )


@pytest.mark.parametrize(
    "batchio", [AudioSamples, PrecomputedFeatures, partial(OnTheFlyFeatures, Fbank())]
)
@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize(
    "executor_type",
    [
        ThreadPoolExecutor,
        partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")),
    ],
)
def test_batch_io(libri_cut_set, batchio, num_workers, executor_type):
    # does not fail / hang / etc.
    read_fn = batchio(num_workers=num_workers, executor_type=executor_type)
    read_fn(libri_cut_set)


def test_audio_samples_with_custom_field(libri_cut_set):
    batchio = AudioSamples()

    def attach_custom_audio(cut):
        """Simulate adding an additional custom recording"""
        cut.my_favorite_song = cut.recording.perturb_volume(factor=1.1)
        return cut

    # Reject mixed cuts (we don't support mixing custom attributes for now) and add custom audio
    cuts = libri_cut_set.filter(lambda c: isinstance(c, MonoCut)).map(
        attach_custom_audio
    )
    # does not fail / hang / etc.
    audio, audio_lens = batchio(cuts, recording_field="my_favorite_song")
    assert audio.shape[0] == 3

    # check that the audio is not the same as in the default 'recording' field
    audio_default, _ = batchio(cuts)
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(audio, audio_default)


def test_audio_samples_with_missing_custom_field(libri_cut_set):
    batchio = AudioSamples()
    with pytest.raises(AssertionError):
        audio, audio_lens = batchio(libri_cut_set, recording_field="my_favorite_song")


def test_audio_samples_equivalent_to_cut_set_load_audio(libri_cut_set):
    batchio = AudioSamples()
    audio, audio_lens = batchio(libri_cut_set)
    audio2, audio_lens2 = libri_cut_set.load_audio(collate=True)
    np.testing.assert_equal(audio2, audio.numpy())
    np.testing.assert_equal(audio_lens2, audio_lens.numpy())


def test_cut_set_load_audio_collate_false(libri_cut_set):
    audio = libri_cut_set.load_audio()
    assert isinstance(audio, list)


# ----------------------------- AudioSamples Metrics Tests -----------------------------


class TestAudioSamplesMetrics:
    """Tests for AudioSamples latency metrics collection."""

    def test_metrics_disabled_by_default(self):
        loader = AudioSamples()
        assert loader.collect_metrics is False
        assert loader._object_latencies is None
        assert loader._batch_latencies is None

    def test_metrics_enabled_via_flag(self):
        loader = AudioSamples(collect_metrics=True)
        assert loader.collect_metrics is True
        assert loader._object_latencies == []
        assert loader._batch_latencies == []

    @patch.dict(os.environ, {"LHOTSE_METRICS_DIR": "/tmp/test_metrics"})
    def test_metrics_enabled_via_env(self):
        loader = AudioSamples()
        assert loader.collect_metrics is True
        assert loader._metrics_dir == "/tmp/test_metrics"

    def test_get_metrics_raises_when_disabled(self):
        loader = AudioSamples()
        with pytest.raises(RuntimeError, match="Metrics collection is not enabled"):
            loader.get_metrics()

    def test_get_metrics_raises_when_no_data(self):
        loader = AudioSamples(collect_metrics=True)
        with pytest.raises(RuntimeError, match="No metrics data collected yet"):
            loader.get_metrics()

    def test_per_object_and_batch_latencies_collected(self, libri_cut_set):
        loader = AudioSamples(collect_metrics=True)
        loader(libri_cut_set)

        # libri_cut_set has 4 cuts, so we expect 4 per-object latencies
        assert len(loader._object_latencies) == 4
        assert all(t > 0 for t in loader._object_latencies)

        # One batch call
        assert len(loader._batch_latencies) == 1
        batch_time, num_objects = loader._batch_latencies[0]
        assert batch_time > 0
        assert num_objects == 4

    def test_get_metrics_structure(self, libri_cut_set):
        loader = AudioSamples(collect_metrics=True)
        loader(libri_cut_set)

        metrics = loader.get_metrics()

        # Check per_object structure
        assert "per_object" in metrics
        for key in ("p50", "p95", "p99", "avg"):
            assert key in metrics["per_object"]
            assert isinstance(metrics["per_object"][key], float)
            assert metrics["per_object"][key] > 0

        # Check batch structure
        assert "batch" in metrics
        for key in ("p50", "p95", "p99", "avg", "per_object_avg"):
            assert key in metrics["batch"]
            assert isinstance(metrics["batch"][key], float)
            assert metrics["batch"][key] > 0

        assert metrics["num_batches"] == 1
        assert metrics["total_objects"] == 4

    def test_metrics_accumulate_across_calls(self, libri_cut_set):
        loader = AudioSamples(collect_metrics=True)
        loader(libri_cut_set)
        loader(libri_cut_set)

        assert len(loader._batch_latencies) == 2
        assert len(loader._object_latencies) == 8  # 4 cuts x 2 calls

        metrics = loader.get_metrics()
        assert metrics["num_batches"] == 2
        assert metrics["total_objects"] == 8

    def test_reset_metrics(self, libri_cut_set):
        loader = AudioSamples(collect_metrics=True)
        loader(libri_cut_set)

        assert len(loader._object_latencies) == 4
        assert len(loader._batch_latencies) == 1

        loader.reset_metrics()

        assert loader._object_latencies == []
        assert loader._batch_latencies == []

    def test_save_metrics(self, libri_cut_set, tmp_path):
        loader = AudioSamples(collect_metrics=True)
        loader(libri_cut_set)

        loader.save_metrics(str(tmp_path), rank=0)

        files = list(tmp_path.glob("rank_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert "per_object" in data
        assert "batch" in data
        assert data["rank"] == 0
        assert "pid" in data

    def test_metrics_with_executor(self, libri_cut_set):
        loader = AudioSamples(num_workers=2, collect_metrics=True)
        loader(libri_cut_set)

        assert len(loader._object_latencies) == 4
        assert all(t > 0 for t in loader._object_latencies)
        assert len(loader._batch_latencies) == 1

    def test_no_overhead_when_disabled(self, libri_cut_set):
        loader = AudioSamples(collect_metrics=False)
        loader(libri_cut_set)

        assert loader._object_latencies is None
        assert loader._batch_latencies is None
