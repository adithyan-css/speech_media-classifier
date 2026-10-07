"""Tests for :mod:`aacara.train` against TRD section 8.

The TRD 8 components — loss weighting, balanced sampling, optimiser, schedule, run
metadata, checkpointing, early stopping — are tested on tiny synthetic tensors and run in
milliseconds.

The last test is the PHASE_PLAN A11 definition of done: 100 windows from the A2 fixture
driven to near-zero loss. It takes about a minute, because it is a real training run with
the TRD 8 hyper-parameters rather than a shortcut.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from aacara.config import load_config  # noqa: E402
from aacara.features.descriptors import N_DESCRIPTORS  # noqa: E402
from aacara.models import cnn  # noqa: E402
from aacara.train import (  # noqa: E402
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    POSITIVE_FRACTION,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    RunMetadata,
    WindowDataset,
    balanced_sampler,
    build_optimizer,
    build_scheduler,
    fit,
    fixture_windows,
    overfit_fixture,
    pos_weight_from_labels,
    set_seed,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


def tiny_dataset(n: int = 40, positives: int = 10, n_mels: int = 8, n_frames: int = 12, seed: int = 0):
    """A small dataset with the right shapes, for the mechanics rather than the learning."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=np.float32)
    labels[:positives] = 1.0
    features = rng.standard_normal((n, 1, n_mels, n_frames)).astype(np.float32)
    features[labels == 1] += 2.0
    descriptors = rng.standard_normal((n, N_DESCRIPTORS)).astype(np.float32)
    return WindowDataset(features, descriptors, labels)


class TinyModel(nn.Module):
    """A stand-in with the TRD 5.4 output contract: one logit, no sigmoid."""

    def __init__(self, n_mels: int = 8, n_frames: int = 12) -> None:
        super().__init__()
        self.linear = nn.Linear(n_mels * n_frames + N_DESCRIPTORS, 1)

    def forward(self, x, descriptors=None):
        flat = torch.flatten(x, 1)
        if descriptors is None:
            descriptors = flat.new_zeros((flat.shape[0], N_DESCRIPTORS))
        return self.linear(torch.cat([flat, descriptors], dim=1))


# --------------------------------------------------------------------------------------
# The TRD 8 hyper-parameters
# --------------------------------------------------------------------------------------


def test_hyperparameters_match_the_trd():
    assert LEARNING_RATE == 3e-4
    assert WEIGHT_DECAY == 1e-4
    assert BATCH_SIZE == 128
    assert EPOCHS == 60
    assert WARMUP_EPOCHS == 5
    assert MIN_LEARNING_RATE == 1e-5
    assert EARLY_STOP_PATIENCE == 10
    assert POSITIVE_FRACTION == 0.30


# --------------------------------------------------------------------------------------
# Loss weighting (TRD 8)
# --------------------------------------------------------------------------------------


def test_pos_weight_is_the_negative_positive_ratio():
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    assert float(pos_weight_from_labels(labels)) == pytest.approx(3.0)


def test_pos_weight_needs_a_positive():
    with pytest.raises(ValueError, match="no positive windows"):
        pos_weight_from_labels(np.zeros(10, dtype=np.float32))


def test_loss_uses_logits_not_probabilities():
    """BCEWithLogitsLoss, so the model must not apply a sigmoid (TRD 5.4, TRD 8)."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
    confident_right = criterion(torch.tensor([[10.0]]), torch.tensor([[1.0]]))
    confident_wrong = criterion(torch.tensor([[-10.0]]), torch.tensor([[1.0]]))
    assert float(confident_right) < 1e-3 < float(confident_wrong)


# --------------------------------------------------------------------------------------
# Balanced sampling (TRD 8: ~30% positive windows per batch)
# --------------------------------------------------------------------------------------


def test_balanced_sampler_reaches_the_target_positive_fraction():
    labels = np.concatenate([np.ones(20), np.zeros(980)]).astype(np.float32)
    sampler = balanced_sampler(labels, POSITIVE_FRACTION)

    set_seed(0)
    drawn = np.asarray([labels[index] for index in sampler])
    assert drawn.mean() == pytest.approx(POSITIVE_FRACTION, abs=0.05)


def test_balanced_sampler_oversamples_a_rare_class():
    """The raw split is 2% positive; the sampler must lift it to ~30%."""
    labels = np.concatenate([np.ones(20), np.zeros(980)]).astype(np.float32)
    assert labels.mean() == pytest.approx(0.02)

    set_seed(1)
    drawn = np.asarray([labels[index] for index in balanced_sampler(labels)])
    assert drawn.mean() > 10 * labels.mean()


def test_balanced_sampler_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        balanced_sampler(np.ones(10, dtype=np.float32))


# --------------------------------------------------------------------------------------
# Optimiser and schedule (TRD 8)
# --------------------------------------------------------------------------------------


def test_optimiser_is_adamw_with_the_trd_settings():
    optimizer = build_optimizer(TinyModel())
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == LEARNING_RATE
    assert optimizer.param_groups[0]["weight_decay"] == WEIGHT_DECAY


def test_schedule_warms_up_linearly_then_anneals():
    optimizer = build_optimizer(TinyModel())
    scheduler = build_scheduler(optimizer, epochs=EPOCHS)

    optimizer.step()        # silences the scheduler-before-optimizer warning
    rates = []
    for _ in range(EPOCHS):
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Five epochs of linear warm-up: 1/5, 2/5, ... 5/5 of the base rate.
    warmup = rates[:WARMUP_EPOCHS]
    assert warmup == pytest.approx(
        [LEARNING_RATE * (i + 1) / WARMUP_EPOCHS for i in range(WARMUP_EPOCHS)]
    )
    assert warmup[-1] == pytest.approx(LEARNING_RATE)

    # Then a monotone cosine decay down to the floor.
    anneal = rates[WARMUP_EPOCHS:]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(anneal, anneal[1:]))
    assert rates[-1] < LEARNING_RATE * 0.05


def test_schedule_floor_is_the_trd_minimum():
    optimizer = build_optimizer(TinyModel())
    scheduler = build_scheduler(optimizer, epochs=20)
    optimizer.step()
    for _ in range(20):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(MIN_LEARNING_RATE, rel=1e-6)


# --------------------------------------------------------------------------------------
# Run metadata (TRD 8: config hash, git SHA, dataset hash, seed)
# --------------------------------------------------------------------------------------


def test_run_metadata_records_what_the_trd_requires(config):
    dataset = tiny_dataset()
    metadata = RunMetadata.build(config, dataset, seed=7, model_name="TinyModel")

    assert metadata.seed == 7
    assert len(metadata.config_hash) == 16
    assert len(metadata.dataset_hash) == 16
    assert metadata.git_sha
    assert metadata.model == "TinyModel"


def test_dataset_hash_changes_with_the_data(config):
    first = RunMetadata.build(config, tiny_dataset(seed=0), 1, "m").dataset_hash
    same = RunMetadata.build(config, tiny_dataset(seed=0), 1, "m").dataset_hash
    other = RunMetadata.build(config, tiny_dataset(seed=1), 1, "m").dataset_hash
    assert first == same
    assert first != other


def test_config_hash_changes_with_the_config(config):
    import dataclasses

    dataset = tiny_dataset()
    baseline = RunMetadata.build(config, dataset, 1, "m").config_hash
    changed = dataclasses.replace(
        config, decision=dataclasses.replace(config.decision, theta_on=0.7)
    )
    assert RunMetadata.build(changed, dataset, 1, "m").config_hash != baseline


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


def test_fit_reduces_the_training_loss(config):
    dataset = tiny_dataset()
    history = fit(TinyModel(), dataset, config=config, epochs=30, batch_size=16, seed=1)
    assert len(history.train_loss) == 30
    assert history.train_loss[-1] < history.train_loss[0]


def build_seeded(seed: int):
    """Seed before constructing, so weight initialisation is covered too.

    ``fit`` seeds the loop - sampler order and dropout masks - but the weights are already
    initialised by the time it is called, so a run is only reproducible end to end if the
    caller seeds first. ``overfit_fixture`` does this.
    """
    set_seed(seed)
    return TinyModel()


def test_fit_is_reproducible_for_a_fixed_seed(config):
    dataset = tiny_dataset()
    first = fit(build_seeded(3), dataset, config=config, epochs=10, batch_size=16, seed=3)
    second = fit(build_seeded(3), dataset, config=config, epochs=10, batch_size=16, seed=3)
    assert first.train_loss == pytest.approx(second.train_loss)


def test_different_seeds_differ(config):
    dataset = tiny_dataset()
    first = fit(build_seeded(3), dataset, config=config, epochs=10, batch_size=16, seed=3)
    second = fit(build_seeded(4), dataset, config=config, epochs=10, batch_size=16, seed=4)
    assert first.train_loss != pytest.approx(second.train_loss)


def test_seeding_the_loop_alone_does_not_fix_weight_init(config):
    """Documents the boundary: same fit seed, unseeded construction, different runs."""
    dataset = tiny_dataset()
    set_seed(99)
    first = fit(TinyModel(), dataset, config=config, epochs=5, batch_size=16, seed=3)
    second = fit(TinyModel(), dataset, config=config, epochs=5, batch_size=16, seed=3)
    assert first.train_loss != pytest.approx(second.train_loss)


def test_early_stopping_fires_on_a_stalled_dev_score(config):
    """A dev score that never improves must stop the run after ``patience`` epochs."""
    dataset = tiny_dataset()
    history = fit(
        TinyModel(), dataset, dataset, config=config, epochs=60, batch_size=16, seed=1,
        patience=3, score_fn=lambda model, data: 0.0,
    )
    assert history.stopped_early
    assert len(history.train_loss) == 4, "one improving epoch, then three without"
    assert history.selection_metric == "dev_event_f1"


def test_no_early_stop_when_the_score_keeps_improving(config):
    dataset = tiny_dataset()
    counter = {"value": 0.0}

    def improving(model, data):
        counter["value"] += 1.0
        return counter["value"]

    history = fit(
        TinyModel(), dataset, dataset, config=config, epochs=12, batch_size=16, seed=1,
        patience=3, score_fn=improving,
    )
    assert not history.stopped_early
    assert len(history.train_loss) == 12
    assert history.best_epoch == 11


def test_selection_metric_falls_back_to_dev_loss(config):
    """TRD 8 selects on dev event-level F1; without a dev split the history says so."""
    dataset = tiny_dataset()
    history = fit(TinyModel(), dataset, dataset, config=config, epochs=5, batch_size=16, seed=1)
    assert history.selection_metric == "dev_loss"
    assert len(history.dev_score) == 5


def test_checkpoints_are_written(config, tmp_path):
    """TRD 8: best dev score, plus last."""
    dataset = tiny_dataset()
    history = fit(
        TinyModel(), dataset, dataset, config=config, epochs=6, batch_size=16, seed=1,
        checkpoint_dir=tmp_path,
    )
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "history.json").is_file()

    payload = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["seed"] == 1
    assert len(payload["train_loss"]) == 6
    assert payload["best_epoch"] == history.best_epoch


def test_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        WindowDataset(np.zeros((4, 1, 8, 8)), np.zeros((3, N_DESCRIPTORS)), np.zeros(4))


def test_fit_works_with_the_real_cnn(config):
    """The loop must drive the TRD 5.1 model, not only the stand-in."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((4, 1, 64, 298)).astype(np.float32)
    descriptors = rng.standard_normal((4, N_DESCRIPTORS)).astype(np.float32)
    labels = np.array([1, 0, 1, 0], dtype=np.float32)
    dataset = WindowDataset(features, descriptors, labels)

    history = fit(cnn.build(config), dataset, config=config, epochs=2, batch_size=2, seed=1)
    assert len(history.train_loss) == 2
    assert all(np.isfinite(history.train_loss))


# --------------------------------------------------------------------------------------
# PHASE_PLAN A11 definition of done
# --------------------------------------------------------------------------------------


def test_fixture_windows_are_labelled_and_shaped(config):
    dataset = fixture_windows(n_windows=10, seconds=120.0)
    assert len(dataset) == 10
    assert dataset.features.shape[1:] == (1, 64, 298)
    assert dataset.descriptors.shape[1] == N_DESCRIPTORS
    assert set(dataset.labels.tolist()) == {0.0, 1.0}, "both classes must be present"


@pytest.mark.slow
def test_overfits_one_hundred_fixture_windows_to_near_zero_loss():
    """The A11 definition of done.

    100 windows from the A2 fixture, the TRD 8 optimiser and schedule, no shortcuts. This
    proves the loop is wired — gradients flow, the sampler feeds, the loss is on the right
    side of the logit — and says nothing about generalisation.

    Takes roughly a minute: it is a real training run.
    """
    history, final_loss = overfit_fixture(n_windows=100, epochs=250, seed=1)

    assert final_loss < 0.05, "eval-mode loss {:.4f} is not near zero".format(final_loss)
    assert history.train_loss[-1] < history.train_loss[0] / 100.0
    assert all(np.isfinite(history.train_loss))
