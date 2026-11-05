# Super SloMo: reproduction report (DRAFT, 3 Nov 2026)

> **Status:** the implementation, tests and data-free checks are complete. Every section marked **PENDING** needs the pretrained checkpoint and/or Vimeo-90K, which were not available when this draft was written. No number in this draft comes from the pretrained model or from Vimeo-90K.

## 1. Goal and scope

This project validates and reuses the Super SloMo frame interpolator (Jiang et al., CVPR 2018) together with the public pretrained weights from `avinashpaliwal/Super-SloMo`, and compares it against two classical optical-flow baselines (Lucas–Kanade and Farnebäck). It also includes a small, controlled fine-tune of the refinement network on Vimeo-90K triplets. It is **not** a full retraining, and its numbers are not comparable to the paper's.

## 2. Implementation

| Component | What is implemented |
| --- | --- |
| Flow U-Net | `cat(I0, I1)` (6 ch) → `F_01 = out[:, 0:2]`, `F_10 = out[:, 2:4]` |
| Intermediate flow (eq. 4) | `F_t0 = −t(1−t)F_01 + t²F_10`, `F_t1 = (1−t)²F_01 − t(1−t)F_10` |
| Refinement U-Net | 20 ch in, in the order `I0, I1, F_01, F_10, F_t1, F_t0, g(I1,F_t1), g(I0,F_t0)`. 5 ch out: `ΔF_t0, ΔF_t1`, visibility logit |
| Visibility | `V_t0 = σ(logit)`, `V_t1 = 1 − V_t0` |
| Fusion | `I_t = ((1−t)V_t0 g(I0,F_t0') + t V_t1 g(I1,F_t1')) / ((1−t)V_t0 + tV_t1 + ε)` |
| Warping | `grid_sample`, bilinear, zero padding |
| Pre/post-processing | RGB, mean (0.429, 0.431, 0.397), std 1; pad to a multiple of 32, crop back; clamp to [0, 1] |
| Multi-frame | flow net once per pair, refinement net once per t; predictions are never fed back |

Both U-Nets have the reference architecture: 7×7 convs, an encoder of avg-pool + conv blocks, a decoder of bilinear upsampling + skip connections, and leaky-ReLU(0.1) on every layer, including the output layer. Parameter counts are 19.79 M (flow) and 19.82 M (refinement). A reduced-width "tiny" variant exists only for tests and the synthetic demo. It is never loaded with pretrained weights.

**Deviations from the reference and why**
1. **Padding instead of resizing.** The reference video script resizes frames to a multiple of 32. Here frames are replicate-padded and then cropped, which avoids resampling.
2. **Warp normalisation.** The reference normalises coordinates as `2(x/W − 0.5)` under the old `align_corners=True` default. That is a sub-pixel shrink, not an exact identity. The default here is the exact form (`2x/(W−1) − 1`). The legacy form can be selected with `--warp-mode legacy`. The choice between them is made empirically (§3).
3. **ε in the fusion denominator** (1e-8), to avoid 0/0.
4. **Fine-tuning loss.** The fine-tune uses L1 reconstruction + warping + smoothness, but no perceptual (VGG) term, since that would need a weight download. With the flow net frozen, only the reconstruction term carries gradient.
5. **Baselines.** These are not in the paper. LK and Farnebäck flows are converted to `F_t0`/`F_t1` with the same eq. 4, warped with `cv2.remap` and blended with weights (1−t, t), without any occlusion handling.

## 3. Checkpoint compatibility evidence: **PENDING**

What has been verified so far:
- Strict loading of a checkpoint in the reference layout (keys `state_dictFC`/`state_dictAT`, 46/46 tensors per net, all shapes equal). This was tested with a randomly initialised full-width model.
- The loader handles the `module.` prefix, reports missing keys clearly, and fails loudly when the tiny model is given full-width weights.

To complete: run `python scripts/check_checkpoint.py --checkpoint checkpoints/SuperSloMo.ckpt --frame0 … --frame1 … --compare-warp-modes`, paste its key/shape summary and metadata here, and include `outputs/pretrained/check/side_by_side.png`.

## 4. Setup and compute

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GiB VRAM, driver 592.82 |
| OS | Windows 11 (10.0.26200) |
| Software | Python 3.12.1, PyTorch 2.6.0+cu126 (CUDA 12.6, cuDNN 9.5), OpenCV 4.11, scikit-image 0.26 |
| Memory check | Full-width fine-tune with frozen flow net, 256×448, batch 6, AMP: fits in 4 GiB |
| Wall-clock | fine-tune: PENDING; evaluation: PENDING |

## 5. Exact commands

See `README.md` (main commands). The commands actually run so far:

```bash
python -m pytest -q
python scripts/smoke_test.py --device cpu --out outputs/smoke
python scripts/interpolate.py --method lk        --input outputs/smoke/smoke.mp4 --factor 4 --out outputs/lk/smoke_4x.mp4
python scripts/interpolate.py --method farneback --input outputs/smoke/smoke.mp4 --factor 4 --out outputs/farneback/smoke_4x.mp4
python scripts/evaluate.py --synthetic 30 --methods lk farneback average --times 0.25 0.5 0.75 --num-comparisons 3 --out outputs/synthetic_eval
python scripts/compare.py --synthetic 3 --methods lk farneback average superslomo --out outputs/synthetic_eval
```

## 6. Fine-tuning hyperparameters: **PENDING** (`runs/finetune/config.json`)

The planned settings are: 700 train and 100 val triplets drawn only from `tri_trainlist.txt` (seed 0, disjoint subsets), 2 epochs, Adam with lr 1e-5, batch 6, AMP, gradient clipping at 1.0, flow U-Net frozen, flip and temporal-reverse augmentation.

## 7. Results

### 7.1 Vimeo-90K: **PENDING** (`outputs/eval/summary.json`)

| Dataset / t | Method | PSNR (dB) | SSIM | N |
| --- | --- | --- | --- | --- |
| triplet test, t = 0.5 | LK / Farnebäck / pretrained / fine-tuned | – | – | – |
| septuplet test, t = 0.25 / 0.5 / 0.75 | … | – | – | – |

### 7.2 Synthetic pipeline check (not a benchmark)

There are 30 rendered scenes (a textured background, a translating checkerboard and an occluding bar moving the opposite way). Each scene is evaluated at t = 0.25/0.5/0.75, so N = 90 frames per method.

| Method | PSNR (dB) | SSIM | N | Runtime/frame |
| --- | --- | --- | --- | --- |
| Lucas–Kanade | 18.45 | 0.794 | 90 | 19 ms (CPU) |
| Farnebäck | 18.44 | 0.785 | 90 | 12 ms (CPU) |
| Frame averaging | 18.51 | 0.791 | 90 | <1 ms |

On these scenes the classical methods do not beat frame averaging. The large disoccluded regions and the periodic stripe texture break brightness-constancy flow, and neither baseline reasons about occlusion. This makes the synthetic scenes a useful stress test, but it says nothing about real video. On a pure-translation pair (unit test), both baselines beat averaging by more than 3 dB.

## 8. Qualitative analysis: **PENDING**

The figures for this section will be `outputs/eval/comparison_<scene>.png` and the best/worst Vimeo samples. The points to discuss are:
- occlusion handling through the visibility maps (the zoom box marks the region where |V_t0 − 0.5| is largest),
- temporal consistency across t,
- failure cases: large motion, thin structures, brightness changes, borders.

## 9. Limitations

- The pretrained weights and Vimeo-90K results are pending (see the status note at the top).
- The fine-tune is small (700 triplets, 2 epochs), the flow net is frozen and there is no perceptual loss. Any change it produces is indicative only.
- Evaluation runs on test-list subsets (`--max-samples`), and each row reports its N. The numbers are not comparable with the paper's, which uses a different training set (Adobe-240fps + YouTube) and full test sets.
- The synthetic scenes are rendered with exact ground truth and are only used to check the pipeline.

## 10. Output file index

| Path | Content |
| --- | --- |
| `outputs/smoke/` | synthetic demo: per-t predictions and GT PNGs, `comparison_smoke.png`, `smoke.mp4` (13 frames) |
| `outputs/lk/smoke_4x.mp4`, `outputs/farneback/smoke_4x.mp4` | 4× slow motion of the smoke clip (49 frames) |
| `outputs/synthetic_eval/` | synthetic `summary.json`, `per_sample.csv`, comparison grids |
| `outputs/pretrained/`, `outputs/eval/`, `runs/finetune/` | PENDING (need the checkpoint and data) |
| `NOTES.md` | phase-by-phase log |
