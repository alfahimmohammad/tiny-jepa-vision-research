"""Minimal I-JEPA: EMA target, 4 target blocks, context block, smooth-L1 on LN targets.

Modified copy of keon/jepa ijepa.py @ d55bab7b (MIT). Only train() changed: step-limited loop on a given dataset,
pluggable mask strategies (models/masks.py) and optional context-input noise. `diff` against upstream to see it.
"""
import copy, math, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def sincos_2d(h, w, dim):
    assert dim % 4 == 0; sub = dim // 4
    yy, xx = [t.reshape(-1).float() for t in torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")]
    div = torch.exp(torch.arange(0, sub * 2, 2).float() * (-math.log(10000.) / (sub * 2)))
    return torch.cat([torch.sin(yy[:, None] * div), torch.cos(yy[:, None] * div),
                      torch.sin(xx[:, None] * div), torch.cos(xx[:, None] * div)], dim=-1)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(self, x):
        h = self.n1(x); x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


def param_groups(modules, wd):
    np_ = [(n, p) for m in modules for n, p in m.named_parameters() if p.requires_grad]
    nd = [p for n, p in np_ if p.ndim < 2 or n.endswith("bias")]
    d = [p for n, p in np_ if p.ndim >= 2 and not n.endswith("bias")]
    return [{"params": d, "weight_decay": wd}, {"params": nd, "weight_decay": 0.0}]


@torch.no_grad()
def ema_update(tgt, online, m):
    # theta_bar <- m * theta_bar + (1 - m) * theta
    for pt, po in zip(tgt.parameters(), online.parameters()): pt.mul_(m).add_(po.detach(), alpha=1 - m)


def lr_warmup_cosine(step, total, base, warmup_frac=0.05):
    warm = max(1, int(total * warmup_frac))
    if step < warm: return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Encoder(nn.Module):                                 # f_theta (context encoder)
    def __init__(self, img_size=32, patch_size=4, in_chans=3, dim=128, depth=6, heads=4):
        super().__init__()
        self.grid = img_size // patch_size; self.n_patches = self.grid ** 2
        self.dim = dim; self.patch_size = patch_size; self.img_size = img_size
        self.patch_proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("pos", sincos_2d(self.grid, self.grid, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, imgs, idx=None):
        tokens = self.patch_proj(imgs).flatten(2).transpose(1, 2)
        B, N, D = tokens.shape
        if idx is None:
            idx = torch.arange(N, device=imgs.device).expand(B, -1); x = tokens + self.pos[idx]
        else:
            x = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)) + self.pos[idx]
        for blk in self.blocks: x = blk(x)
        return self.norm(x)


class Predictor(nn.Module):                              # g_phi
    def __init__(self, grid, enc_dim=128, dim=64, depth=4, heads=4):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, dim); self.out_proj = nn.Linear(dim, enc_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sincos_2d(grid, grid, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, ctx, ctx_idx, tgt_idx):
        B, T = ctx.size(0), tgt_idx.size(1)
        x = torch.cat([self.in_proj(ctx) + self.pos[ctx_idx],
                       self.mask_token.expand(B, T, -1) + self.pos[tgt_idx]], dim=1)
        for blk in self.blocks: x = blk(x)
        return self.out_proj(self.norm(x[:, -T:]))


def _bsize(g, s, ar):
    a = s * g * g
    return (max(1, min(g, round(math.sqrt(a * ar)))), max(1, min(g, round(math.sqrt(a / ar)))))


def _block(g, top, left, h, w):
    return {r * g + c for r in range(top, top + h) for c in range(left, left + w)}


def sample_ijepa_masks(B, grid, n_targets=4, min_ctx=4, rng=None):
    """Block sizes shared per batch; locations per item; random-subsample trim."""
    rng = rng or random
    th, tw = _bsize(grid, rng.uniform(0.15, 0.20), rng.uniform(0.75, 1.5))
    ch, cw = _bsize(grid, rng.uniform(0.85, 1.0), 1.0)
    ctx_list = [None] * B; tgt_lists = [[None] * B for _ in range(n_targets)]
    for b in range(B):
        ts = []
        for m in range(n_targets):
            top, left = rng.randint(0, grid - th), rng.randint(0, grid - tw)
            t = _block(grid, top, left, th, tw); ts.append(t); tgt_lists[m][b] = sorted(t)
        for _ in range(10):
            ct, cl = rng.randint(0, grid - ch), rng.randint(0, grid - cw)
            c = _block(grid, ct, cl, ch, cw) - set().union(*ts)
            if len(c) >= min_ctx: break
        ctx_list[b] = sorted(c) if c else [0]
    L = min(len(c) for c in ctx_list)
    return [sorted(rng.sample(c, L)) for c in ctx_list], tgt_lists


def add_context_noise(imgs, gen, p=0.5, sigma_max=0.2):
    """Noise on the CONTEXT-encoder input only (the target encoder keeps the clean image): each image is noised
    with probability p by N(0, sigma^2), sigma ~ U(0, sigma_max), in [0, 1] pixel space, clipped, re-normalised."""
    mean = torch.tensor(MEAN, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=imgs.device).view(1, 3, 1, 1)
    apply = torch.rand(imgs.size(0), 1, 1, 1, generator=gen, device=imgs.device) < p
    sigma = torch.rand(imgs.size(0), 1, 1, 1, generator=gen, device=imgs.device) * sigma_max * apply
    noise = torch.randn(imgs.shape, generator=gen, device=imgs.device)
    noisy = ((imgs * std + mean + sigma * noise).clamp(0, 1) - mean) / std
    return torch.where(apply, noisy, imgs)


def train(dataset, steps=500, batch_size=64, lr=3e-4, wd=0.05, ema_start=0.996, ema_end=1.0,
          mask_fn=None, ctx_noise=None, seed=0, num_workers=2, device=None, on_eval=None, eval_every=0):
    """Step-limited version of the original epoch loop. Changes: a fixed number of optimizer steps over `dataset`
    (reshuffled with a seeded generator every epoch); `mask_fn(B, grid, rng)` replaces sample_ijepa_masks
    (models/masks.py); optional
    `ctx_noise` = dict(p, sigma_max, seed) adds noise to the context-encoder input; `on_eval(step, ctx_enc,
    tgt_enc, pred)` runs at step 0 and every `eval_every` steps (excluded from the training time)."""
    device = device or pick_device(); print(f"device: {device}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mask_fn = mask_fn or (lambda B, grid, rng: sample_ijepa_masks(B, grid, rng=rng))

    def loader(epoch):                                  # new seeded shuffle every epoch
        g = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
        return iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                               generator=g, worker_init_fn=lambda w: np.random.seed((seed * 7919 + epoch * 31 + w) % 2**32)))

    ctx_enc = Encoder().to(device); tgt_enc = copy.deepcopy(ctx_enc).to(device)
    for p in tgt_enc.parameters(): p.requires_grad_(False)
    pred = Predictor(grid=ctx_enc.grid).to(device)
    init = {"ctx_enc": copy.deepcopy(ctx_enc.state_dict()), "predictor": copy.deepcopy(pred.state_dict())}
    opt = torch.optim.AdamW(param_groups([ctx_enc, pred], wd), lr=lr)
    rng = random.Random(seed); log = []; D = ctx_enc.dim; epoch = 0; it = loader(0); eval_time = 0.0
    noise_gen = torch.Generator(device=device).manual_seed(ctx_noise["seed"] + seed) if ctx_noise else None

    def run_eval(step):
        nonlocal eval_time
        t = time.time(); on_eval(step, ctx_enc, tgt_enc, pred); eval_time += time.time() - t

    if on_eval is not None: run_eval(0)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(steps):
        try: imgs = next(it)
        except StopIteration: epoch += 1; it = loader(epoch); imgs = next(it)
        imgs = imgs.to(device)
        cl, tls = mask_fn(imgs.size(0), ctx_enc.grid, rng)
        ci = torch.tensor(cl, device=device)
        tis = [torch.tensor(t, device=device) for t in tls]
        for g in opt.param_groups: g["lr"] = lr_warmup_cosine(step, steps, lr)
        ctx_imgs = add_context_noise(imgs, noise_gen, ctx_noise["p"], ctx_noise["sigma_max"]) if ctx_noise else imgs
        with torch.no_grad(): raw = tgt_enc(imgs); full = F.layer_norm(raw, (D,))  # LN(s_y); no_grad = stop-gradient
        ce = ctx_enc(ctx_imgs, ci)                                          # s_x = f_theta(x_context)
        loss = sum(
            F.smooth_l1_loss(
                pred(ce, ci, ti),                                           # hat_s_y(i) = g_phi(s_x, B_i)
                full.gather(1, ti.unsqueeze(-1).expand(-1, -1, D)))         # [LN(s_y)]_{B_i}
            for ti in tis                                                   # for i in 1..M
        ) / len(tis)                                                        # (1/M) * sum
        opt.zero_grad(); loss.backward()
        gnorm = torch.sqrt(sum(p.grad.pow(2).sum() for mod in (ctx_enc, pred) for p in mod.parameters()
                               if p.grad is not None)).item()                    # logged only (no clipping)
        opt.step()
        m = ema_start + (ema_end - ema_start) * (step / max(1, steps - 1))
        ema_update(tgt_enc, ctx_enc, m)
        log.append({"step": step, "loss": loss.item(), "lr": opt.param_groups[0]["lr"], "ema": m,
                    "ctx_tokens": ci.size(1), "tgt_tokens": sum(t.size(1) for t in tis), "grad_norm": gnorm,
                    "ctx_len": ci.size(1),                                           # = ctx_tokens (notebook name)
                    "tgt_union": float(np.mean([len(set().union(*(t[b] for t in tls))) for b in range(len(cl))])),
                    "tgt_pooled_std": raw.mean(1).std(0).mean().item(),             # target-encoder pooled features
                    "ctx_pooled_std": ce.detach().mean(1).std(0).mean().item()})
        if not math.isfinite(log[-1]["loss"]): raise FloatingPointError(f"non-finite loss at step {step}")
        if step % 50 == 0:
            print(f"ep={epoch} step={step:5d} loss={loss.item():.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} ema={m:.4f} ctx={ci.size(1)}", flush=True)
        if on_eval is not None and eval_every and ((step + 1) % eval_every == 0 or step + 1 == steps):
            run_eval(step + 1)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return {"ctx_enc": ctx_enc, "tgt_enc": tgt_enc, "predictor": pred, "init": init, "log": log,
            "train_time_s": time.time() - t0 - eval_time, "device": device,
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None}
