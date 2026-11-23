"""The target encoder gets no loss gradients and changes only through the EMA update."""
import torch

from models.modified_ijepa import ema_update, train


def _ds(n=16):
    g = torch.Generator().manual_seed(0)
    return [torch.randn(3, 32, 32, generator=g) for _ in range(n)]


def test_one_step_target_is_exact_ema():
    m = 0.9
    out = train(_ds(), steps=1, batch_size=8, ema_start=m, ema_end=m, num_workers=0, device="cpu")
    init, ctx_after, tgt_after = out["init"]["ctx_enc"], out["ctx_enc"].state_dict(), out["tgt_enc"].state_dict()
    changed = 0
    for k, v in tgt_after.items():
        if k == "pos":
            continue
        assert torch.allclose(v, m * init[k] + (1 - m) * ctx_after[k], atol=1e-6), k
        changed += int(not torch.equal(v, init[k]))
    assert changed > 0
    assert all(not p.requires_grad and p.grad is None for p in out["tgt_enc"].parameters())


def test_m_equal_one_freezes_target():
    out = train(_ds(), steps=2, batch_size=8, ema_start=1.0, ema_end=1.0, num_workers=0, device="cpu")
    for k, v in out["tgt_enc"].state_dict().items():
        assert torch.equal(v, out["init"]["ctx_enc"][k])


def test_ema_update_formula():
    a, b = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    before = [p.clone() for p in a.parameters()]
    ema_update(a, b, 0.75)
    for p0, p, q in zip(before, a.parameters(), b.parameters()):
        assert torch.allclose(p, 0.75 * p0 + 0.25 * q)
