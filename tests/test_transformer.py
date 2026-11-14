"""Smoke tests for the transformer pipeline.

From repo root, run:
    python -m test.test_transformer_module

Tests 1-8 run on CPU with random tensors (no dataset, no GPU).
Test 9 composes real Hydra config the way train.py does.
Test 10 compares max_len with real sequence lengths and needs the data/ folder
(it is skipped if the files are missing).
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from emg2qwerty.charset import charset
from emg2qwerty.lightning import TransformerCTCModule
from emg2qwerty.modules import TransformerEncoder

torch.manual_seed(0)

# Checked against the repo: config/model/tds_conv_ctc.yaml (in_features, mlp_features),
# config/transforms/log_spectrogram.yaml (n_fft=64 gives 33 frequency bins) and
# config/decoder/ctc_greedy.yaml. Test 9 re-checks them against model config.
IN_FEATURES = 528                      # front end input: 33 freq bins x 16 channels
FREQ_BINS = 33
MLP_FEATURES = [384]
NUM_FEATURES = 2 * MLP_FEATURES[-1]    # 768: front end output = encoder input width
DECODER = OmegaConf.create({"_target_": "emg2qwerty.decoder.CTCGreedyDecoder"})

# Filename (without .yaml) of model config in config/model/. None skips tests 9 and 10.
MODEL_CONFIG = "transformer_ctc"

NUM_CLASSES = charset().num_classes
T, N, TARGET_LEN = 100, 2, 10


def build_encoder(**overrides):
    kwargs = dict(
        num_classes=NUM_CLASSES, num_features=NUM_FEATURES, d_model=256,
        nhead=4, num_layers=4, dim_ff=1024, dropout=0.1, max_len=5000,
    )
    kwargs.update(overrides)
    return TransformerEncoder(**kwargs)


def build_module(**overrides):
    # optimizer and lr_scheduler are only stored here; test 9 builds them from the real config
    return TransformerCTCModule(
        in_features=IN_FEATURES,
        mlp_features=MLP_FEATURES,
        optimizer=OmegaConf.create({}),
        lr_scheduler=OmegaConf.create({}),
        decoder=DECODER,
        **overrides,
    )


def fake_batch():
    return {
        "inputs": torch.randn(T, N, 2, 16, FREQ_BINS),
        "targets": torch.randint(0, charset().null_class, (TARGET_LEN, N)),
        "input_lengths": torch.full((N,), T),
        "target_lengths": torch.full((N,), TARGET_LEN),
    }


# ---------- Encoder on its own (input is the front end's output) ----------

# 1. Shape, NaNs, log-probabilities, parameter count
enc = build_encoder().eval()
x = torch.randn(T, N, NUM_FEATURES)
with torch.no_grad():
    out = enc(x)
assert out.shape == (T, N, NUM_CLASSES), out.shape
assert not torch.isnan(out).any()
assert torch.allclose(out.exp().sum(-1), torch.ones(T, N), atol=1e-4)
print("1. encoder OK, params:", sum(p.numel() for p in enc.parameters()))

# 2. Permutation test: shuffling time steps must change the output
perm = torch.randperm(T)
with torch.no_grad():
    same = torch.allclose(enc(x[perm]), out[perm], atol=1e-5)
print("2. permutation-equivariant (want False):", same)

# 3. Padding test: padded frames must not affect valid frames
lengths = torch.tensor([T, T // 2])
kpm = torch.arange(T)[None, :] >= lengths[:, None]       # (N, T), True = padding
x2 = x.clone()
for n in range(N):
    x2[int(lengths[n]):, n] = torch.randn(T - int(lengths[n]), NUM_FEATURES)
with torch.no_grad():
    a = enc(x, key_padding_mask=kpm)
    b = enc(x2, key_padding_mask=kpm)
for n in range(N):
    assert torch.allclose(a[: int(lengths[n]), n], b[: int(lengths[n]), n], atol=1e-5)
print("3. padding test passed")

# ---------- The Lightning module ----------

# 4. Settings reach the encoder and are saved as hyperparameters
module = build_module(d_model=128, nhead=2, num_layers=2, dim_ff=512, max_len=3000)
inner = module.model[-1]
assert module.hparams.d_model == 128 and module.hparams.num_layers == 2
assert len(inner.encoder.layers) == 2
assert inner.positional_encoding.pe.shape == (3000, 1, 128)
print("4. settings wired through, params:", sum(p.numel() for p in module.parameters()))

# 5. Full model forward on raw-shaped input
module = build_module().eval()
with torch.no_grad():
    out = module(fake_batch()["inputs"])
assert out.shape == (T, N, NUM_CLASSES), out.shape
assert not torch.isnan(out).any()
print("5. module forward OK:", tuple(out.shape), "| params:",
      sum(p.numel() for p in module.parameters()))

# 6. One training step: finite loss, and every parameter gets a gradient
# (a self.log warning here is expected: the module is not attached to a Trainer)
module.train()
loss = module._step("train", fake_batch())
assert torch.isfinite(loss), loss
loss.backward()
missing = [n for n, p in module.named_parameters() if p.grad is None]
print("6. loss:", round(loss.item(), 3), "| params without grad:", missing or "none")

# 7. Overfit one batch: the loss should fall steadily
module = build_module(dropout=0.0).train()
batch = fake_batch()
opt = torch.optim.AdamW(module.parameters(), lr=3e-4)
losses = []
for step in range(151):
    opt.zero_grad()
    loss = module._step("train", batch)
    loss.backward()
    nn.utils.clip_grad_norm_(module.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
    if step % 50 == 0:
        print(f"7. step {step}: loss {loss.item():.3f}")
verdict = "OK" if losses[-1] < 0.5 * losses[0] else "did not halve: try more steps or check the wiring"
print(f"7. loss {losses[0]:.3f} -> {losses[-1]:.3f} ({verdict})")

# 8. Informational: what happens past the positional table length?
module.eval()
pos = getattr(module.model[-1], "positional_encoding", None)
if pos is None or not hasattr(pos, "pe"):
    print("8. skipped (no fixed positional table found)")
else:
    max_len = pos.pe.shape[0]
    try:
        with torch.no_grad():
            module(torch.randn(max_len + 1, 1, 2, 16, FREQ_BINS))
        print("8. longer-than-max_len input did NOT fail (unexpected)")
    except RuntimeError as e:
        print(f"8. T > max_len ({max_len}) fails as expected:", str(e)[:70])

# ---------- Tests real config ----------

if MODEL_CONFIG is None:
    print("9-10. skipped (set MODEL_CONFIG to test Hydra config)")
else:
    from hydra import compose, initialize
    from hydra.utils import instantiate

    # 9. Compose the config like train.py and build the module the way train.py does.
    # The model YAML only holds `module:` and `datamodule:`. optimizer, lr_scheduler and
    # decoder come from base.yaml's defaults list, and train.py passes them in as keyword
    # arguments, so we do the same here.
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="base", overrides=[f"model={MODEL_CONFIG}"])

    n_fft, hop = cfg.logspec.n_fft, cfg.logspec.hop_length
    assert cfg.module.in_features == IN_FEATURES, cfg.module.in_features
    assert list(cfg.module.mlp_features) == MLP_FEATURES, list(cfg.module.mlp_features)
    assert cfg.module.in_features == (n_fft // 2 + 1) * 16, "in_features must match n_fft"

    module = instantiate(
        cfg.module,
        optimizer=cfg.optimizer,
        lr_scheduler=cfg.lr_scheduler,
        decoder=cfg.decoder,
        _recursive_=False,
    )
    print("9. config OK; configure_optimizers keys:", list(module.configure_optimizers()))

    # 10. max_len versus real sequence lengths (frames, not raw samples)
    dm = cfg.datamodule
    train_frames = (dm.window_length + sum(dm.padding) - n_fft) // hop + 1
    max_len = cfg.module.get("max_len")
    print(f"10. train/val frames per window: {train_frames} | max_len in config: {max_len}")

    test_frames = []
    try:
        import h5py

        for s in cfg.dataset.test:
            path = Path("data") / f"{s.session}.hdf5"
            if path.exists():
                with h5py.File(path, "r") as f:
                    n_samples = f["emg2qwerty"]["timeseries"].shape[0]   # samples at 2 kHz
                test_frames.append((n_samples - n_fft) // hop + 1)
    except ImportError:
        pass

    if max_len is None:
        print("10. no fixed max_len in the config (positional encoding computed on the fly?)")
    else:
        assert max_len >= train_frames, f"max_len {max_len} < train frames {train_frames}"
        if test_frames:
            longest = max(test_frames)
            print(f"10. longest test session: {longest} frames ({longest / 125 / 60:.1f} min)")
            assert max_len >= longest, f"max_len {max_len} < longest test session {longest}"
            print("10. max_len covers training windows and the test session")
        else:
            print("10. test session files not found in data/: only the training length was checked")