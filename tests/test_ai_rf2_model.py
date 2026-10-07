"""Tests for metrics, the model wrapper, SpecAugment and the corpus plan."""
from __future__ import annotations

import collections

import numpy as np
import pytest
import torch

from ai_rf2 import corpus, metrics
from ai_rf2.model import SpeechMusicNet, descriptor_stats, load_checkpoint, save_checkpoint
from ai_rf2.train import spec_augment


def test_threshold_for_far_bounds_false_alarms():
    rng = np.random.default_rng(0)
    neg = rng.normal(-2, 1, 1000)
    pos = rng.normal(2, 1, 1000)
    logits = np.r_[neg, pos]
    labels = np.r_[np.zeros(1000), np.ones(1000)]
    for far in (0.01, 0.05, 0.2):
        thr = metrics.threshold_for_far(logits, labels, far)
        op = metrics.at_threshold(logits, labels, thr)
        assert op["far"] <= far + 1e-12
        assert op["far"] >= far - 0.002          # and it is not needlessly strict


def test_window_metrics_perfect_and_chance():
    labels = np.r_[np.zeros(50), np.ones(50)]
    perfect = metrics.window_metrics(np.r_[-np.ones(50), np.ones(50)], labels)
    assert perfect["auc"] == 1.0 and perfect["dr_at_far5"] == 1.0
    assert perfect["accuracy@0.5"] == 1.0
    chance = metrics.window_metrics(np.zeros(100), labels)
    assert chance["auc"] == 0.5


def test_model_standardises_descriptors_and_ignores_constant_ones():
    desc = np.random.default_rng(1).normal(1000, 50, (64, 8)).astype(np.float32)
    desc[:, 7] = 1.0                                   # stereo correlation: constant in the PoC
    mean, std = descriptor_stats(desc)
    torch.manual_seed(0)
    model = SpeechMusicNet("cnn", mean, std).eval()
    assert model.desc_std[7].item() == 1.0             # zero-variance guarded
    mel = torch.randn(4, 1, 64, 298)
    out = model(mel, torch.from_numpy(desc[:4]))
    assert out.shape == (4, 1) and torch.all(torch.isfinite(out))


def test_model_parameter_count_matches_trd():
    model = SpeechMusicNet("cnn")
    assert sum(p.numel() for p in model.net.parameters()) == 81_793   # TRD 5.1


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = SpeechMusicNet("cnn", np.arange(8, dtype=np.float32), np.ones(8, np.float32)).eval()
    save_checkpoint(tmp_path / "m.pt", model, {"note": "x"})
    loaded, extra = load_checkpoint(tmp_path / "m.pt")
    mel, desc = torch.randn(2, 1, 64, 298), torch.randn(2, 8)
    assert torch.allclose(model(mel, desc), loaded(mel, desc))
    assert extra == {"note": "x"}


def test_spec_augment_masks_within_limits():
    x = torch.ones(8, 1, 64, 298)
    y = spec_augment(x.clone(), torch.Generator().manual_seed(0))
    for i in range(8):
        zero_frames = int((y[i, 0].sum(0) == 0).sum())
        zero_bands = int((y[i, 0].sum(1) == 0).sum())
        assert zero_frames <= 60 and zero_bands <= 16


@pytest.mark.skipif(not (corpus.ROOT / "data" / "sources.csv").exists(), reason="no manifest")
def test_manifest_groups_never_cross_splits():
    rows = corpus.read_sources()
    splits = collections.defaultdict(set)
    for r in rows:
        splits[r["group"]].add(r["split"])
    assert all(len(v) == 1 for v in splits.values())
    assert {r["label"] for r in rows} == {"SPEECH", "MEDIA"}
    # every language except Russian is in train; Russian is test-only (unseen language)
    langs_train = {r["language"] for r in rows if r["split"] == "train" and r["label"] == "SPEECH"}
    langs_test = {r["language"] for r in rows if r["split"] == "test" and r["label"] == "SPEECH"}
    assert "ru" not in langs_train and "ru" in langs_test


def test_threshold_max_margin_centres_between_limits():
    rng = np.random.default_rng(3)
    logits = np.r_[rng.normal(-3, 1, 5000), rng.normal(3, 1, 5000)]
    labels = np.r_[np.zeros(5000), np.ones(5000)]
    thr = metrics.threshold_max_margin(logits, labels)
    op = metrics.at_threshold(logits, labels, thr)
    # symmetric classes: both margins close to each other, both acceptance limits met
    assert op["dr"] >= 0.95 and op["far"] <= 0.05
    assert abs((op["dr"] - 0.95) - (0.05 - op["far"])) < 0.01
    assert abs(thr) < 0.2
