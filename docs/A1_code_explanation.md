# A1 — How upstream `ijepa.py` works

Line numbers refer to `ijepa.py` in [keon/jepa](https://github.com/keon/jepa) at commit
`d55bab7b712d4dfa8bcfb1afc9196e454a56dde7`. Measured numbers come from
`scripts/mask_stats.py` → `results/mask_stats.json`.

## 1. From image to patch tokens, and what the `Encoder` sees

* **Patchify + embed (L58, L64).** `patch_proj` is a `Conv2d(3, 128, kernel=4, stride=4)`. On a
  32×32 image it gives a 128×8×8 map; `flatten(2).transpose(1, 2)` turns it into **64 tokens of
  dimension 128**. Token `p` is patch `(row, col) = divmod(p, 8)` (row-major), the same indexing
  the mask sampler uses.
* **Position (L59, L10–15).** A fixed (not learned) 2-D sin-cos table `pos` of shape (64, 128):
  32 dims each for sin/cos of the row and of the column. It is a buffer, so neither the optimizer
  nor the EMA touches it.
* **Dropping masked patches (L66–69).** With `idx=None` all 64 tokens are used. With an index
  tensor `idx` (B, L), the encoder `gather`s only those tokens, adds *their* positions, and runs the
  transformer on an **L-token sequence**. Masked patches are physically removed: there is no mask
  token in the encoder, so attention cannot reach them. The convolution runs over the whole image
  first, but because kernel = stride = 4, each token depends only on its own 4×4 pixels, so
  gathering afterwards does not leak target pixels (checked in
  `tests/test_model.py::test_context_encoder_never_sees_target_tokens`).
* **Transformer (L18–27, L70–71).** 6 pre-LayerNorm blocks (`x + MHA(LN(x))`, `x + MLP(LN(x))`,
  4 heads, MLP ratio 4), then a final LayerNorm. There is no CLS token; downstream we mean-pool the
  64 output tokens.

## 2. How `sample_ijepa_masks()` builds context and target indices (L91–117)

1. **One block shape per batch (L103–104).** Target shape: scale ~ U(0.15, 0.20) of the 64-patch
   grid, aspect ratio ~ U(0.75, 1.5). `_bsize` turns that into an integer (h, w), clamped to
   [1, 8]. Context shape: scale ~ U(0.85, 1.0), aspect ratio 1. Measured: a target block covers
   9–16 patches (mean 11.3).
2. **Per image, place 4 target blocks (L108–110)** with uniformly random top-left corners. They
   may overlap each other. On average their union covers about 32 of the 64 patches.
3. **Per image, place the context block and remove every target patch from it (L111–114).** It
   retries up to 10 times until at least `min_ctx=4` patches survive. If the context is still empty,
   it **falls back to `[0]` (L115)**, and patch 0 can be a target patch. That is a latent leak,
   impossible in practice at grid 8 (0 of 128k sampled images) but reachable on tiny grids. Our
   training loop wraps the sampler with `check_masks` and resamples if it ever happens
   (`models/ijepa_base.py::sample_masks_checked`).
4. **Trim to a rectangular batch (L116–117).** Every image's context is randomly subsampled down to
   the **shortest context in the batch**, `L = min_b |ctx_b|`. Targets are not trimmed: they
   already share one size.

Consequences that matter for the hypotheses:

* **Visible context is small.** At batch size 64 the trimmed context has about **20 of 64 patches
  (~31%)**. From that, the predictor must produce **4 × ~11 ≈ 45 target tokens**. In ~3.5% of
  batches the context drops below 10 patches.
* **The random trim breaks the context's contiguity** (it becomes a sparse random subset of a
  block).
* **Context size depends on batch size.** The trim is a minimum over the batch, so larger batches
  give smaller contexts (`results/mask_stats.json → by_batch_size`). Our batch of 64 therefore
  defines a slightly easier task than upstream's default of 256. Any Part B change to the sampler
  must keep the per-batch context length and the target-token count matched.

## 3. What the `Predictor` predicts (L74–88), and why in latent space

* Input: the context encoder's output `s_x` (B, L, 128), its indices, and **one** target block's
  indices `B_i` (B, T).
* `in_proj` maps 128 → 64 and adds the 64-d sin-cos position of each context patch. For each target
  patch it adds one shared learnable `mask_token` plus that patch's position. The concatenated
  sequence `[context ; queries]` passes through 4 transformer blocks with full attention. The last T
  outputs go through LN and `out_proj` 64 → 128.
* So the predictor says: **"given these visible features, here is the target encoder's feature at
  position (r, c)."** The target is `LN(f_θ̄(x))` at those positions (L143, L148), where LN
  normalises each token over its 128 dims. The predictor is called once per target block (L145–149),
  so the 4 blocks never attend to each other.
* **Why features, not pixels.** A pixel loss makes the model spend capacity on exact texture,
  lighting and noise, much of which cannot be predicted from distant context and does not matter
  for recognition. Predicting the target encoder's representation lets the model choose an
  abstraction where unpredictable detail can be dropped. Large masked blocks (rather than single
  patches) push it toward longer-range, more semantic regularities. The price is that the target is
  produced by the network itself, so a trivial solution exists (see 4–5). It also does not need the
  hand-designed view augmentations of contrastive methods; upstream still applies RandomResizedCrop
  (L123), as the official I-JEPA code does.

## 4. `ema_update()` and the stop-gradient (L37–40, L127–153)

* `tgt_enc` is a `deepcopy` of the context encoder at initialisation with `requires_grad=False`
  (L127–128). Its forward pass runs under `torch.no_grad()` (L143). The optimizer only receives the
  context-encoder and predictor parameters (L130). So the loss has **no gradient path** into the
  target encoder (stop-gradient).
* After each optimizer step, `ema_update` sets θ̄ ← m·θ̄ + (1−m)·θ for every parameter (L152–153),
  with m rising linearly from 0.996 to 1.0 over training. The target encoder is therefore a slow,
  lagged average of the context encoder. Near the end m → 1, so it effectively freezes.
* This asymmetry (stop-grad + EMA + a predictor on one branch only) is what keeps the jointly
  learned target from immediately collapsing. Nothing in the loss explicitly forbids collapse:
  there is no variance or covariance term. The LayerNorm on targets (L143) fixes each token's scale,
  but **a constant vector shared by all images and positions survives LN**, so LN does not prevent
  collapse.
* **This matters at a 500-step budget.** Under a linear schedule from 0.996 the target keeps a
  weight of ∏ m_s ≈ e^{-1} ≈ **0.37 on its random initialisation** after 500 steps (≈0.13 after
  1,000, ≈0 after 5,000; see `results/mask_stats.json`). Over most of a short run the regression
  target is substantially a random network's features. Upstream's schedule was meant for long
  training.

## 5. Low loss vs. collapse vs. poor downstream accuracy

These are three different measurements and can disagree:

| Outcome | What it means | How we measure it |
|---|---|---|
| **Low prediction loss** | The predictor matches `LN(f_θ̄)` at target positions. The target itself moves with training, so loss values are not comparable across models with different target encoders. | `heldout_loss` on fixed val images and masks |
| **Representation collapse** | Features barely vary across images: complete (constant output) or dimensional (only a few directions vary). Complete collapse makes the loss trivially ~0. | per-feature std across images, effective rank, top singular-value share, mean pairwise cosine |
| **Poor downstream accuracy** | Features vary but do not linearly separate classes, e.g. they encode position, colour statistics or texture rather than object identity. | frozen linear probe on clean and corrupted test images |

Concrete ways the loss can be low without useful features in this code:

* **Positional shortcut.** The predictor is told the target positions, and the targets contain
  position information (sin-cos is added at the input, L67/L69). If the target features are
  dominated by position or image-generic structure, the predictor can match them without using the
  image. Measured by `heldout_loss_shuffled_ctx` (context taken from another image) and
  `heldout_loss_mean_target` (predict the batch-average target at each position). If these are close
  to the real loss, the predictor is not really using the content.
* **Easy targets.** A near-random target encoder, or a slowly moving one (see 4), produces smooth,
  low-rank targets that are easy to regress but carry little class information.
* **Mean pooling plus a 2,000-image probe** can hide information that is present but not linearly
  available after averaging. That would be an evaluation limitation, not a representation failure.

## 6. Where we modify the code, and what must stay invariant

| Code path | Change for this assignment | Invariant to preserve |
|---|---|---|
| `train()` loop (L120–162) | Re-implemented in `train.py` as a fixed-step loop on the 10k subset, batch 64, LR and EMA schedules over total steps, checkpoints, logging | Same loss, optimizer, param groups, schedules, augmentation; EMA applied **after** `opt.step()`; target not in the optimizer |
| Loss (L143–150) | Lifted verbatim into `ijepa_loss` (tested equal to the inline code) | Smooth-L1 on LN target features, average over M=4 blocks, target under `no_grad` |
| `sample_ijepa_masks` (L100–117) | Part B target: mask geometry | Valid indices; context ∩ targets = ∅; rectangular batch; **matched** context length and target-token count when comparing geometries |
| `Encoder.forward` `idx` path (L63–71) | none | Context encoder only receives visible tokens; evaluation calls it with `idx=None` |
| `ema_update` (L37–40) | none | θ̄ ← mθ̄ + (1−m)θ, no gradients into θ̄ |
| Evaluation (new `evaluate.py`) | Frozen **context** encoder (upstream extras probe the target encoder), mean pooling, fixed probe protocol | Same splits, probe schedule, selection on validation only, identical corruption tensors (hash-checked) for every model |
