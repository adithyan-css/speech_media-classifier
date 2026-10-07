"""Training loop (TRD section 8).

Every hyper-parameter below is TRD 8's, named and gathered at the top of the module so a
change is visible. They are *not* in ``configs/default.yaml``: that file is the contract
between the training and inference planes (TRD 1), and the only part of it the runtime
consumes is the ``decision:`` block. Training settings never cross that boundary.

    python -m aacara.train --overfit --seed 1

**Definition of done for this module is the overfit smoke test** (PHASE_PLAN A11): 100
windows drawn from the A2 synthetic fixture must be driven to near-zero loss. That proves
the loop is wired — gradients flow, the sampler feeds, the scheduler steps, the loss is
on the right side of the logit — without any claim that the model generalises.

Not yet wired, and blocked rather than stubbed:

* the manifest-driven dataset (A7) and the per-file mel cache (A13) do not exist, so
  :func:`fit` takes tensors in memory rather than a split name;
* early stopping is on **dev event-level F1** per TRD 8, which needs the evaluation
  harness over a dev split; :func:`fit` accepts a ``score_fn`` for it and falls back to
  dev loss when none is given, saying so in the history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from aacara.config import Config, load_config

__all__ = [
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "BATCH_SIZE",
    "EPOCHS",
    "WARMUP_EPOCHS",
    "MIN_LEARNING_RATE",
    "EARLY_STOP_PATIENCE",
    "POSITIVE_FRACTION",
    "WindowDataset",
    "RunMetadata",
    "set_seed",
    "pos_weight_from_labels",
    "balanced_sampler",
    "build_optimizer",
    "build_scheduler",
    "fit",
    "overfit_fixture",
    "main",
]

# --- TRD 8 --------------------------------------------------------------------------
LEARNING_RATE = 3e-4            #: AdamW, lr 3e-4
WEIGHT_DECAY = 1e-4             #: AdamW, weight decay 1e-4
BATCH_SIZE = 128                #: 128 windows
EPOCHS = 60                     #: 60 epochs
WARMUP_EPOCHS = 5               #: 5-epoch linear warm-up
MIN_LEARNING_RATE = 1e-5        #: cosine anneal to 1e-5
EARLY_STOP_PATIENCE = 10        #: early stop on dev event-level F1, patience 10
POSITIVE_FRACTION = 0.30        #: balanced sampler, ~30% positive windows per batch


class WindowDataset(Dataset):
    """Log-Mel windows, their auxiliary descriptors and their labels.

    Args:
        features: ``(n, 1, n_mels, n_frames)`` float32.
        descriptors: ``(n, 8)`` float32 (TRD 4).
        labels: ``(n,)`` float32, 1 for TA and 0 for MEDIA.
    """

    def __init__(self, features: np.ndarray, descriptors: np.ndarray, labels: np.ndarray) -> None:
        if not (len(features) == len(descriptors) == len(labels)):
            raise ValueError("features, descriptors and labels must have the same length")
        self.features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        self.descriptors = torch.as_tensor(np.asarray(descriptors), dtype=torch.float32)
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32).reshape(-1)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.features[index], self.descriptors[index], self.labels[index]


@dataclass
class RunMetadata:
    """What TRD 8 requires every run to record: config hash, git SHA, dataset hash, seed."""

    seed: int
    config_hash: str
    dataset_hash: str
    git_sha: str
    model: str

    @staticmethod
    def build(config: Config, dataset: WindowDataset, seed: int, model_name: str) -> "RunMetadata":
        return RunMetadata(
            seed=seed,
            config_hash=_hash_object(asdict_config(config)),
            dataset_hash=_hash_tensor(dataset.features, dataset.labels),
            git_sha=_git_sha(),
            model=model_name,
        )


def asdict_config(config: Config) -> dict:
    """The config as a plain nested dict, for hashing."""
    import dataclasses

    return dataclasses.asdict(config)


def _hash_object(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _hash_tensor(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()[:16]


def _git_sha() -> str:
    """The current commit, or ``"unknown"`` when this is not a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def set_seed(seed: int) -> None:
    """Fix every RNG the loop touches (TRD 8: seeds fixed and logged)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------------------
# Loss, sampling, optimiser, schedule
# --------------------------------------------------------------------------------------


def pos_weight_from_labels(labels: torch.Tensor | np.ndarray) -> torch.Tensor:
    """TRD 8: ``pos_weight`` is the train-split negative/positive ratio."""
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32).reshape(-1)
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    if positives == 0:
        raise ValueError("cannot weight a split with no positive windows")
    return torch.tensor([negatives / positives], dtype=torch.float32)


def balanced_sampler(
    labels: torch.Tensor | np.ndarray, positive_fraction: float = POSITIVE_FRACTION
) -> WeightedRandomSampler:
    """TRD 8: a balanced sampler drawing ~30% positive windows per batch.

    Each class is given a total sampling mass equal to its target share, so the expected
    positive fraction per batch is ``positive_fraction`` regardless of the split's own
    class balance.
    """
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32).reshape(-1)
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("a balanced sampler needs both classes")

    weights = torch.where(
        labels == 1,
        torch.full_like(labels, positive_fraction / positives),
        torch.full_like(labels, (1.0 - positive_fraction) / negatives),
    )
    return WeightedRandomSampler(weights.double(), num_samples=len(labels), replacement=True)


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """TRD 8: AdamW, lr 3e-4, weight decay 1e-4."""
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int = EPOCHS,
    warmup_epochs: int = WARMUP_EPOCHS,
    min_lr: float = MIN_LEARNING_RATE,
) -> torch.optim.lr_scheduler.LambdaLR:
    """TRD 8: 5-epoch linear warm-up, then cosine anneal to 1e-5. Stepped per epoch."""
    floor = min_lr / LEARNING_RATE

    def factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        span = max(epochs - warmup_epochs, 1)
        progress = min((epoch - warmup_epochs) / span, 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


@dataclass
class History:
    """Per-epoch record of a run."""

    metadata: RunMetadata
    train_loss: list[float] = field(default_factory=list)
    dev_score: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = float("-inf")
    stopped_early: bool = False
    selection_metric: str = "dev_loss"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["metadata"] = asdict(self.metadata)
        return payload


def fit(
    model: nn.Module,
    train_dataset: WindowDataset,
    dev_dataset: WindowDataset | None = None,
    *,
    config: Config | None = None,
    seed: int = 1,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOP_PATIENCE,
    balanced: bool = True,
    score_fn=None,
    checkpoint_dir: str | Path | None = None,
    device: str = "cpu",
) -> History:
    """Train to TRD 8 and return the per-epoch history.

    Args:
        model: any of the TRD 5 models; must take ``(x, descriptors)`` and return logits.
        train_dataset: the training windows.
        dev_dataset: optional; drives early stopping and checkpoint selection.
        score_fn: ``(model, dev_dataset) -> float``, higher is better. TRD 8 selects on
            dev **event-level F1**; pass that here once a dev split exists. Without it,
            selection falls back to negative dev loss and ``selection_metric`` says so.
        checkpoint_dir: when given, ``best.pt`` and ``last.pt`` are written there
            (TRD 8: best dev score, plus last).

    Returns:
        A :class:`History`, including the run metadata TRD 8 requires.
    """
    config = config if config is not None else load_config()
    # Seeds the loop: sampler order and dropout masks. Weight initialisation happened
    # when the caller built the model, so seed before constructing it for a run that is
    # reproducible end to end - :func:`overfit_fixture` does exactly that.
    set_seed(seed)
    model = model.to(device)

    sampler = balanced_sampler(train_dataset.labels) if balanced else None
    loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=False,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from_labels(train_dataset.labels))
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, epochs=epochs)

    history = History(
        metadata=RunMetadata.build(config, train_dataset, seed, type(model).__name__),
        selection_metric="dev_event_f1" if score_fn is not None else "dev_loss",
    )
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    since_improvement = 0
    for epoch in range(epochs):
        model.train()
        total, seen = 0.0, 0
        for features, descriptors, labels in loader:
            features = features.to(device)
            descriptors = descriptors.to(device)
            labels = labels.to(device).reshape(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features, descriptors), labels)
            loss.backward()
            optimizer.step()

            total += float(loss.detach()) * features.shape[0]
            seen += features.shape[0]

        history.train_loss.append(total / max(seen, 1))
        history.learning_rate.append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()

        if dev_dataset is not None:
            score = (
                float(score_fn(model, dev_dataset))
                if score_fn is not None
                else -evaluate_loss(model, dev_dataset, criterion, device)
            )
            history.dev_score.append(score)
            if score > history.best_score:
                history.best_score = score
                history.best_epoch = epoch
                since_improvement = 0
                if checkpoint_dir is not None:
                    torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            else:
                since_improvement += 1
                if since_improvement >= patience:
                    history.stopped_early = True
                    break

    if checkpoint_dir is not None:
        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        (checkpoint_dir / "history.json").write_text(
            json.dumps(history.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    return history


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, dataset: WindowDataset, criterion: nn.Module, device: str = "cpu"
) -> float:
    """Mean loss over a dataset, in eval mode."""
    model.eval()
    logits = model(dataset.features.to(device), dataset.descriptors.to(device))
    return float(criterion(logits, dataset.labels.to(device).reshape(-1, 1)))


# --------------------------------------------------------------------------------------
# The overfit smoke test (PHASE_PLAN A11 definition of done)
# --------------------------------------------------------------------------------------


def fixture_windows(
    n_windows: int = 100, seconds: float = 120.0, seed: int = 20260919, config: Config | None = None
) -> WindowDataset:
    """Draw labelled windows from the A2 synthetic fixture.

    Half TA and half MEDIA, taken from the fixture's own RDS timeline, so the smoke test
    has both classes without needing a manifest (A7) or a mel cache (A13).
    """
    import json as _json
    import tempfile
    import wave

    from aacara.features.descriptors import compute as compute_descriptors
    from aacara.features.frontend import to_work_rate
    from aacara.features.melspec import window_tensor
    from tests.fixtures.make_synthetic import generate

    config = config if config is not None else load_config()
    directory = Path(tempfile.mkdtemp(prefix="aacara_overfit_"))
    generate(directory, seconds=seconds, seed=seed, config=config)

    with wave.open(str(directory / "fixture.wav"), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    interleaved = np.frombuffer(raw, dtype="<i2")
    mono = to_work_rate(interleaved, config)
    timeline = _json.loads((directory / "fixture_rds.json").read_text(encoding="utf-8"))

    rate = config.audio.sample_rate_work
    span = int(round(config.features.window_seconds * rate))
    hop = int(round(config.features.window_hop_seconds * rate))

    def label_at(start_sample: int) -> int:
        centre = (start_sample + span / 2) / rate
        return int(any(entry["start_s"] <= centre < entry["end_s"] for entry in timeline))

    starts = list(range(0, mono.size - span, hop))
    positives = [s for s in starts if label_at(s) == 1]
    negatives = [s for s in starts if label_at(s) == 0]

    rng = np.random.default_rng(seed)
    half = n_windows // 2
    chosen = list(rng.choice(positives, size=min(half, len(positives)), replace=False))
    chosen += list(rng.choice(negatives, size=n_windows - len(chosen), replace=False))

    features, descriptors, labels = [], [], []
    for start in chosen:
        window = mono[start:start + span]
        features.append(window_tensor(window, config))
        descriptors.append(compute_descriptors(window, config))
        labels.append(label_at(int(start)))

    return WindowDataset(np.stack(features), np.stack(descriptors), np.asarray(labels))


def overfit_fixture(
    n_windows: int = 100,
    epochs: int = 120,
    seed: int = 1,
    batch_size: int = 50,
    dataset: WindowDataset | None = None,
) -> tuple[History, float]:
    """Drive ``n_windows`` fixture windows to near-zero loss (PHASE_PLAN A11).

    Sampling is unbalanced here on purpose: the point is to memorise this exact set, not
    to train a usable model.

    Returns:
        ``(history, final_eval_loss)`` with the loss measured in eval mode.
    """
    from aacara.models import cnn

    config = load_config()
    dataset = dataset if dataset is not None else fixture_windows(n_windows, config=config)

    set_seed(seed)              # before the model is built, so weight init is seeded too
    model = cnn.build(config)
    history = fit(
        model, dataset, config=config, seed=seed, epochs=epochs,
        batch_size=batch_size, balanced=False,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from_labels(dataset.labels))
    return history, evaluate_loss(model, dataset, criterion)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aacara.train",
        description="Training loop (TRD 8). Only the overfit smoke test is wired so far.",
    )
    parser.add_argument("--overfit", action="store_true", help="run the A11 smoke test")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    if not args.overfit:
        parser.error(
            "manifest-driven training needs A7 (manifest) and A13 (mel cache), which do "
            "not exist yet. Run with --overfit for the A11 smoke test."
        )

    history, final_loss = overfit_fixture(args.windows, args.epochs, args.seed)
    print("seed {}  git {}  config {}  dataset {}".format(
        history.metadata.seed, history.metadata.git_sha,
        history.metadata.config_hash, history.metadata.dataset_hash,
    ))
    print("first epoch loss {:.4f}  last epoch loss {:.6f}  eval loss {:.6f}".format(
        history.train_loss[0], history.train_loss[-1], final_loss,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
