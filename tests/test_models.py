"""Tests for the three model definitions against TRD section 5.

Random tensors only — no training, no data. These tests exist so that a silent
architecture drift fails: every per-layer output shape in the TRD 5.1 table is asserted,
and both the parameter count and the MAC count are checked against the budget.

The MAC count is **computed by a counter in this file**, never hard-coded, so the TRD 11
compute budget stays checkable as the architecture changes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from aacara.config import load_config  # noqa: E402
from aacara.features.descriptors import N_DESCRIPTORS  # noqa: E402
from aacara.models import cnn, crnn, mobilenet  # noqa: E402

# TRD 5.1 budget. The TRD states "approximately 80 k parameters, approximately 11 MMACs".
# TRD 5.1 now states the conventions (conv bias=False, affine BN after both the depthwise
# and the pointwise convolution, linear layers keep biases) and carries the measured
# breakdown, so this is an exact figure rather than a band:
#   conv weights 62,848 + fc weights 16,960 + BN affine 1,920 + fc biases 65 = 81,793
TRD_CNN_PARAMS = 81_793
TRD_CNN_MMACS = 11.0
TRD_CNN_MMAC_TOLERANCE = 0.5

#: Measured from this implementation. Pinned so any architecture change fails visibly,
#: separately from the looser budget check above.
MEASURED = {
    "cnn": (81_793, 10.81),
    "crnn": (226_690, 9.78),
    "mobilenet": (478_401, 21.36),
}


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def tensor_shape(config):
    n_frames = 1 + (
        int(round(config.features.window_seconds * config.audio.sample_rate_work))
        - config.features.win_length
    ) // config.features.hop_length
    return (1, config.features.n_mels, n_frames)


# --------------------------------------------------------------------------------------
# The MAC counter — computed, never asserted from a constant
# --------------------------------------------------------------------------------------


def count_macs(model: nn.Module, x: torch.Tensor) -> int:
    """Multiply-accumulates for one forward pass, counted with forward hooks.

    Convolutions: output elements x input channels per group x kernel area.
    Linear layers: in_features x out_features.
    GRU: steps x directions x 3 gates x (input + hidden) x hidden.

    BatchNorm, activations and pooling are excluded — they are not GEMM work and are not
    what the TRD 11 budget is counting.
    """
    total = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal total
        per_sample = output.numel() / output.shape[0]
        total += (
            per_sample
            * (module.in_channels // module.groups)
            * module.kernel_size[0]
            * module.kernel_size[1]
        )

    def linear_hook(module, inputs, output):
        nonlocal total
        total += module.in_features * module.out_features

    def gru_hook(module, inputs, output):
        nonlocal total
        steps = inputs[0].shape[1]
        directions = 2 if module.bidirectional else 1
        total += steps * directions * 3 * (module.input_size + module.hidden_size) * module.hidden_size

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.GRU):
            handles.append(module.register_forward_hook(gru_hook))

    model.eval()
    with torch.no_grad():
        model(x)
    for handle in handles:
        handle.remove()
    return int(total)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_the_mac_counter_is_correct_on_a_known_layer():
    """A 3x3 conv, 4->8 channels, over a 10x10 output: 8*10*10 * 4*3*3 = 28800."""
    layer = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
    assert count_macs(layer, torch.zeros(1, 4, 10, 10)) == 8 * 10 * 10 * 4 * 3 * 3


def test_the_mac_counter_handles_depthwise_and_linear():
    depthwise = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False)
    assert count_macs(depthwise, torch.zeros(1, 8, 5, 5)) == 8 * 5 * 5 * 1 * 3 * 3
    assert count_macs(nn.Linear(10, 3), torch.zeros(1, 10)) == 30


# --------------------------------------------------------------------------------------
# TRD 5.1 — the baseline CNN, layer by layer
# --------------------------------------------------------------------------------------


def test_cnn_input_and_output_contract(config, tensor_shape):
    model = cnn.build(config)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, *tensor_shape))
    assert output.shape == (2, 1), "one logit per window (TRD 5.4)"


def test_cnn_layer_shapes_match_the_trd_table(config, tensor_shape):
    """Every row of the TRD 5.1 table, asserted in order."""
    model = cnn.build(config)
    model.eval()
    x = torch.zeros(1, *tensor_shape)
    assert tuple(x.shape[1:]) == (1, 64, 298)

    with torch.no_grad():
        x = model.stem(x)
        assert tuple(x.shape[1:]) == (32, 32, 149), "Conv2d 3x3 1->32 stride 2"
        x = model.block1(x)
        assert tuple(x.shape[1:]) == (64, 16, 75), "DSConv 32->64 stride 2"
        x = model.block2(x)
        assert tuple(x.shape[1:]) == (128, 8, 38), "DSConv 64->128 stride 2"
        x = model.block3(x)
        assert tuple(x.shape[1:]) == (128, 4, 38), "DSConv 128->128 stride (2,1)"
        x = model.block4(x)
        assert tuple(x.shape[1:]) == (256, 2, 19), "DSConv 128->256 stride 2"
        pooled = torch.flatten(model.pool(x), 1)
        assert tuple(pooled.shape[1:]) == (256,), "global average pool"


def test_cnn_concatenates_the_aux_descriptors_at_the_pooling_layer(config, tensor_shape):
    """TRD 5.1: 256 -> 264 after the concat, which is the width fc1 must accept."""
    model = cnn.build(config)
    assert model.fc1.in_features == 256 + N_DESCRIPTORS == 264
    assert model.fc1.out_features == 64
    assert model.fc2.in_features == 64 and model.fc2.out_features == 1


def test_cnn_descriptors_change_the_output(config, tensor_shape):
    """The concat must actually be wired, not accepted and dropped."""
    model = cnn.build(config)
    model.eval()
    x = torch.zeros(2, *tensor_shape)
    with torch.no_grad():
        without = model(x)
        with_values = model(x, torch.ones(2, N_DESCRIPTORS))
    assert not torch.allclose(without, with_values)


def test_cnn_defaults_descriptors_to_zeros(config, tensor_shape):
    model = cnn.build(config)
    model.eval()
    x = torch.zeros(2, *tensor_shape)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x, torch.zeros(2, N_DESCRIPTORS)))


def test_cnn_dropout_probability(config):
    assert cnn.build(config).dropout.p == 0.2


def test_cnn_uses_relu6_for_int8_friendliness(config):
    """ReLU6 bounds the activation range, which is what makes INT8 quantise cleanly."""
    model = cnn.build(config)
    assert any(isinstance(m, nn.ReLU6) for m in model.modules())


def test_cnn_parameter_count_is_exactly_the_trd_figure(config):
    """TRD 5.1 carries the measured breakdown, so this is exact, not a band."""
    assert parameter_count(cnn.build(config)) == TRD_CNN_PARAMS == 81_793


def test_cnn_parameter_breakdown_matches_the_trd(config):
    """The four groups TRD 5.1 tabulates, counted separately so a drift says which moved."""
    model = cnn.build(config)
    conv_weights = sum(
        p.numel() for m in model.modules() if isinstance(m, nn.Conv2d) for p in m.parameters()
    )
    bn_affine = sum(
        p.numel() for m in model.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters()
    )
    fc_weights = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    fc_biases = sum(m.bias.numel() for m in model.modules() if isinstance(m, nn.Linear))

    assert conv_weights == 62_848, "conv bias=False, so weights only"
    assert fc_weights == 16_960
    assert bn_affine == 1_920, "affine BN after both the depthwise and the pointwise conv"
    assert fc_biases == 65, "linear layers keep their biases"
    assert conv_weights + fc_weights + bn_affine + fc_biases == TRD_CNN_PARAMS


def test_cnn_conventions_are_the_ones_trd_51_states(config):
    model = cnn.build(config)
    assert all(m.bias is None for m in model.modules() if isinstance(m, nn.Conv2d))
    assert all(m.affine for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    assert all(m.bias is not None for m in model.modules() if isinstance(m, nn.Linear))


def test_cnn_macs_are_within_the_trd_budget(config, tensor_shape):
    """TRD 5.1: approximately 11 MMACs per window; TRD 11 budgets 12 ms for it."""
    macs = count_macs(cnn.build(config), torch.zeros(1, *tensor_shape)) / 1e6
    assert abs(macs - TRD_CNN_MMACS) <= TRD_CNN_MMAC_TOLERANCE, "measured {:.2f} MMACs".format(macs)
    assert macs == pytest.approx(MEASURED["cnn"][1], abs=0.01) == pytest.approx(10.81, abs=0.01)


def test_cnn_output_is_a_logit_not_a_probability(config, tensor_shape):
    """TRD 5.4: a single logit, so the output range is unbounded.

    Driving the final bias proves it directly. Feeding random input would not: an
    untrained ReLU6 stack saturates to zero, so every window scores the same constant
    and the range says nothing.
    """
    model = cnn.build(config)
    model.eval()
    x = torch.zeros(2, *tensor_shape)

    with torch.no_grad():
        model.fc2.bias.fill_(7.5)
        assert (model(x) > 1.0).all(), "a sigmoid would cap this at 1.0"
        model.fc2.bias.fill_(-7.5)
        assert (model(x) < 0.0).all(), "a sigmoid would floor this at 0.0"
    assert not any(isinstance(m, nn.Sigmoid) for m in model.modules())


def test_cnn_shape_follows_the_config(config, tensor_shape):
    import dataclasses

    narrower = dataclasses.replace(
        config, features=dataclasses.replace(config.features, n_mels=32)
    )
    model = cnn.build(narrower)
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 1, 32, tensor_shape[2])).shape == (2, 1)


# --------------------------------------------------------------------------------------
# TRD 5.2 — the CRNN
# --------------------------------------------------------------------------------------


def test_crnn_output_contract(config, tensor_shape):
    model = crnn.build(config)
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(2, *tensor_shape)).shape == (2, 1)


def test_crnn_reuses_the_first_three_conv_blocks(config, tensor_shape):
    """TRD 5.2 says "first three conv blocks unchanged", so they are imported, not copied."""
    assert isinstance(crnn.build(config).block1, cnn.DepthwiseSeparable)

    model = crnn.build(config)
    model.eval()
    with torch.no_grad():
        x = model.block2(model.block1(model.stem(torch.zeros(1, *tensor_shape))))
    assert tuple(x.shape[1:]) == (128, 8, 38), "the same 128 x 8 x 38 as TRD 5.1 block 3"


def test_crnn_reshapes_to_thirty_eight_by_1024(config):
    """TRD 5.2: reshape 128 x 8 x 38 -> 38 x 1024, then project to 128."""
    model = crnn.build(config)
    assert model.reshaped_width == 128 * 8 == 1024
    assert model.projection.in_features == 1024
    assert model.projection.out_features == crnn.PROJECTION_WIDTH == 128


def test_crnn_gru_is_bidirectional_with_64_units_per_direction(config):
    model = crnn.build(config)
    assert model.gru.bidirectional
    assert model.gru.num_layers == 1
    assert model.gru.hidden_size == crnn.GRU_HIDDEN == 64
    assert model.gru.input_size == 128


def test_crnn_attention_pools_over_time(config, tensor_shape):
    """Attention weights must sum to 1 over the time axis and collapse it."""
    model = crnn.build(config)
    model.eval()
    with torch.no_grad():
        pooled = model.embed(torch.randn(3, *tensor_shape))
    assert pooled.shape == (3, 2 * crnn.GRU_HIDDEN), "time axis is gone"


def test_crnn_head_matches_the_baseline(config):
    model = crnn.build(config)
    assert model.fc1.in_features == 2 * crnn.GRU_HIDDEN + N_DESCRIPTORS
    assert model.fc2.out_features == 1


def test_crnn_parameter_and_mac_counts_are_pinned(config, tensor_shape):
    model = crnn.build(config)
    assert parameter_count(model) == MEASURED["crnn"][0]
    macs = count_macs(model, torch.zeros(1, *tensor_shape)) / 1e6
    assert macs == pytest.approx(MEASURED["crnn"][1], abs=0.01)


def test_crnn_is_larger_than_the_baseline(config):
    """TRD 5.2 puts the CRNN above the baseline in both counts."""
    assert parameter_count(crnn.build(config)) > parameter_count(cnn.build(config))


# --------------------------------------------------------------------------------------
# TRD 5.3 — MobileNetV2, width 0.35
# --------------------------------------------------------------------------------------


def test_mobilenet_output_contract(config, tensor_shape):
    model = mobilenet.build(config)
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(2, *tensor_shape)).shape == (2, 1)


def test_mobilenet_width_multiplier_is_035():
    assert mobilenet.WIDTH_MULT == 0.35


def test_mobilenet_takes_single_channel_input(config, tensor_shape):
    """Adapted from the published 3-channel stem (TRD 5.3)."""
    first_conv = next(m for m in mobilenet.build(config).modules() if isinstance(m, nn.Conv2d))
    assert first_conv.in_channels == 1


def test_mobilenet_uses_the_published_inverted_residual_settings():
    assert mobilenet.SETTINGS == (
        (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
        (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1),
    )


def test_mobilenet_has_residual_connections(config):
    blocks = [m for m in mobilenet.build(config).modules() if isinstance(m, mobilenet.InvertedResidual)]
    assert blocks, "no inverted residual blocks were built"
    assert any(block.use_residual for block in blocks)


def test_mobilenet_head_matches_the_baseline(config):
    model = mobilenet.build(config)
    assert model.fc1.in_features == model.last_channels + N_DESCRIPTORS
    assert model.fc2.out_features == 1


def test_mobilenet_parameter_and_mac_counts_are_pinned(config, tensor_shape):
    model = mobilenet.build(config)
    assert parameter_count(model) == MEASURED["mobilenet"][0]
    macs = count_macs(model, torch.zeros(1, *tensor_shape)) / 1e6
    assert macs == pytest.approx(MEASURED["mobilenet"][1], abs=0.01)


def test_mobilenet_is_the_most_expensive_of_the_three(config):
    """TRD 5.3: included for the ablation table, expected to lose on cost."""
    counts = {
        name: parameter_count(module.build(config))
        for name, module in (("cnn", cnn), ("crnn", crnn), ("mobilenet", mobilenet))
    }
    assert counts["mobilenet"] > counts["crnn"] > counts["cnn"]


# --------------------------------------------------------------------------------------
# Common contract across all three
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("module", [cnn, crnn, mobilenet], ids=["cnn", "crnn", "mobilenet"])
def test_every_model_takes_the_same_input_and_gives_one_logit(module, config, tensor_shape):
    model = module.build(config)
    model.eval()
    torch.manual_seed(1)
    x = torch.randn(4, *tensor_shape)
    with torch.no_grad():
        plain = model(x)
        with_descriptors = model(x, torch.randn(4, N_DESCRIPTORS))
    assert plain.shape == with_descriptors.shape == (4, 1)
    assert torch.isfinite(plain).all()
    assert not torch.allclose(plain, with_descriptors)


@pytest.mark.parametrize("module", [cnn, crnn, mobilenet], ids=["cnn", "crnn", "mobilenet"])
def test_every_model_has_no_sigmoid(module, config):
    assert not any(isinstance(m, nn.Sigmoid) for m in module.build(config).modules())


@pytest.mark.parametrize("module", [cnn, crnn, mobilenet], ids=["cnn", "crnn", "mobilenet"])
def test_every_model_is_trainable(module, config, tensor_shape):
    """A backward pass reaches every parameter — no detached branch, no frozen block."""
    model = module.build(config)
    model.train()
    output = model(torch.randn(2, *tensor_shape), torch.randn(2, N_DESCRIPTORS))
    output.sum().backward()

    without_gradient = [name for name, p in model.named_parameters() if p.grad is None]
    assert not without_gradient, "no gradient reached: {}".format(without_gradient[:5])
