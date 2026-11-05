# NOTES.md: working log (raw material for REPORT.md)

## Environment (recorded 2026-11-03)

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.12.1 (system interpreter, **no venv**: torch+CUDA was already installed; a venv would re-download ~2.5 GB) |
| PyTorch | 2.6.0+cu126, CUDA 12.6, cuDNN 90501 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GiB VRAM, driver 592.82 (nvidia-smi CUDA 13.1) |
| OpenCV | 4.11.0 (`opencv-python` already installed; it provides the same `cv2` as `opencv-python-headless`, so the headless wheel was not installed alongside it to avoid a cv2 package clash) |
| scikit-image | 0.26.0 (installed) |
| pytest | 7.4.4. pytest 9 broke the globally installed `pytest-asyncio 0.23.7` plugin (collection errors), so it was pinned to 7.4.x. This still satisfies `pytest>=7.4`. |

Sanity check: `torch 2.6.0+cu126 cuda True cv2 4.11.0 skimage 0.26.0`.

### Update 2026-11-04: dedicated venv

`requirements.txt` now includes `torch==2.6.0`, pulled from the PyTorch CUDA 12.6 index through `--extra-index-url`. torchvision was left out because no code imports it. A fresh `.venv` was created with `pip install -r requirements.txt`, and it resolved to:

- torch 2.6.0+cu126 (CUDA available on the RTX 3050 Ti)
- numpy 2.5.4
- opencv-python-headless 5.0.0
- scikit-image 0.26.0
- pytest 9.2.0

There is no plugin conflict in the clean venv. Result: `pytest -q` gives 54 passed, 1 skipped.

## Phase 1: baseline health check

**State found:** the folder had no starter code. It contained only `1712.00080v2.pdf` (the paper) and a 3-line `Readme.md`. There was nothing to map onto, so the plan's target layout (section 3) was created from scratch. There was no checkpoint, no Vimeo-90K and no clips.

**Files created**
- `superslomo/model.py`: `UNet` (reference layer names/widths), `backward_warp`, `intermediate_flows`, `SuperSloMo` (`compute_flows`, `synthesize`, `forward`, `interpolate`), `superslomo_loss`.
- `superslomo/preprocess.py`: BGR→RGB I/O, normalise (mean 0.429/0.431/0.397, std 1), denormalise + clamp, pad to /32 (replicate), crop.
- `superslomo/checkpoint.py`: legacy loader/saver, mismatch table.
- `superslomo/data.py`: `VimeoTriplet`, `VimeoSeptuplet`, `SyntheticScene`/`SyntheticMotion`, split helpers.
- `superslomo/baselines.py`, `metrics.py`, `video.py`.
- `superslomo/inference.py` (new, not in plan layout): a single `Interpolator` over all methods plus `interpolate_stream`, so the scripts don't each re-implement method dispatch and the streaming logic can be unit-tested.
- `superslomo/viz.py` (new, not in plan layout): comparison-grid code shared by `compare.py` and `evaluate.py`.
- `scripts/_common.py`: puts the repo root on `sys.path` for `python scripts/x.py`.
- `pytest.ini` (`pythonpath = .`), `.gitignore`, `requirements.txt` (exactly as in the plan).

**Commands:** `python -m pytest -q` → all pass; `python scripts/smoke_test.py --device cpu --out outputs/smoke` → 13 frames, `smoke.mp4`, PSNR table. Runtime was 18 s on an idle CPU and 39 s with the machine under load. The default was reduced from 80 to 30 steps to stay under 1 min.

**Smoke result (synthetic, N = 3 pairs per t, tiny model, 30 steps):** tiny Super SloMo 21.6 dB, average 21.5, Farnebäck 21.8, LK 21.8. A 30-step tiny model is not expected to beat anything. The demo is a pipeline check.

## Phase 2: checkpoint compatibility

**Implementation (matches reference `avinashpaliwal/Super-SloMo`):**
- U-Net: conv7→conv7, down(5×5), 4× down(3×3), 5× up (bilinear ×2 + conv + concat-skip conv), conv3. Leaky-ReLU 0.1 everywhere **including the output layer** (a reference quirk that must be kept for the weights). Avg-pool downsampling. `F.interpolate(..., align_corners=False)`.
- Module attribute names `conv1, conv2, down1..5 (conv1, conv2), up1..5 (conv1, conv2), conv3` give 46 tensors per net, identical key names.
- Parameter counts: flow U-Net 19,794,116; refinement U-Net 19,816,357. A unit test derives the flow count from the reference layer list.
- Refinement input order: `I0, I1, F_01, F_10, F_t1, F_t0, g(I1,F_t1), g(I0,F_t0)`, as in the reference `eval.py`/`video_to_slomo.py`.
- Loader: `weights_only=True`. On failure it raises with a hint to use `--trust-checkpoint`, and only that flag enables `weights_only=False`. It requires `state_dictFC`/`state_dictAT`, strips `module.`, prints a table of missing/unexpected/shape mismatches, and loads strictly.

**Audit checklist (pending the real checkpoint; code-level status):**
1. RGB vs BGR: OpenCV BGR is converted to RGB on load and back on save (`preprocess.load_image_rgb`/`save_image_rgb`). The reference uses PIL, which is RGB. ✔ by construction.
2. Normalisation: mean (0.429, 0.431, 0.397), std 1. The inverse is followed by a clamp to [0, 1]. ✔
3. Refinement channel order: as above. ✔
4. `grid_sample`: the reference `backWarp` normalises `2(x/W − 0.5)` and ran on PyTorch 0.4, whose default was `align_corners=True`. That combination samples at `x·(W−1)/W`, a sub-pixel shrink. Both variants are implemented: `warp_mode="exact"` (default; `2x/(W−1)−1`, `align_corners=True`, so zero flow is the exact identity) and `warp_mode="legacy"` (reference formula). `check_checkpoint.py --compare-warp-modes` reports both. **To do with the real checkpoint:** run it on a real pair and keep the mode that looks or scores better.
5. Padding: replicate-pad bottom/right to /32, then crop. The reference video script *resizes* instead. Padding is the plan's choice and avoids resampling. Warp padding is zeros, as in the reference.
6. Flow channel order: channel 0 = x, channel 1 = y; `out[:, :2] = F_01`, `out[:, 2:] = F_10`; `F_t0`/`F_t1` coefficients copied from the reference. ✔

**Result: the real checkpoint is NOT available** (`checkpoints/SuperSloMo.ckpt` is missing). `check_checkpoint.py` prints this clearly and exits with code 2. The real-checkpoint test is env-gated and **skipped**, not passed. The plumbing was verified with a randomly initialised full-width checkpoint in the reference layout: 46/46 tensors matched per net, strict load OK, the PNGs were written, and exact vs legacy warp differed by 4.5/255 mean. **Compatibility with the pretrained weights is not yet claimed.**

## Phase 3: direct multi-frame inference
- `SuperSloMo.interpolate(I0, I1, ts)` pads once, runs the flow net once and the refinement net once per t, then crops. Tests use forward hooks to check this, including that the refinement inputs are always the original frames (no recursion).
- `interpolate_stream` holds ≤ batch+1 input frames and writes `original, k/N…, original`. A test checks the frame count `N(n−1)+1` for batch sizes 1–3.
- `scripts/interpolate.py`: pair mode and video/dir mode, `--factor`, `--fps`, `--batch-size`, `--fp16`, `--max-frames`, `--warp-mode`. It was verified with LK and a random full checkpoint: 13 frames in → 49 frames out at 4x (`--batch-size 2 --fp16` on GPU).

## Phase 4: baselines and qualitative comparison
- Farnebäck: `calcOpticalFlowFarneback` on grayscale in both directions (levels 5, winsize 15, iterations 5, poly_n 7, sigma 1.5).
- LK: up to 1500 Shi–Tomasi corners, pyramidal LK (21×21, 4 levels), forward–backward check (< 1 px). Densification uses inverse-distance weighting (power 2) of all tracked vectors onto a coarse grid of about 4096 cells, followed by a bilinear resize to full size.
- Both baselines use the **same quadratic intermediate-flow approximation as Super SloMo (paper eq. 4)**, backward warping with `cv2.remap` (border replicate) and a `(1−t, t)` blend with no occlusion reasoning.
- `compare.py` accepts pair folders, Vimeo-style folders and videos. For a video it uses the triplet with the most motion, with the middle frame as GT. It also accepts `--synthetic N`. Each grid has the columns `I0 | LK | Farnebäck | Super SloMo | GT | I1`, a zoom row and an |error| row. The zoom box sits where |V_t0 − 0.5| is largest (the occlusion region per Super SloMo), or at the largest |I1 − I0| when no model is used.
- No real clips are available, so only synthetic comparisons were produced (`outputs/synthetic_eval/comparison_synthetic*.png`). On these synthetic scenes LK/Farnebäck barely match frame averaging: a large occluding bar and a periodic stripe texture defeat classical flow. These scenes are a pipeline check, not evidence about real video.

## Phase 5: fine-tuning
- `scripts/finetune.py`: it checks its preconditions and exits with code 3 and `FINE-TUNING SKIPPED: <reason>` when CUDA, Vimeo or the checkpoint is missing. Train/val subsets come only from `tri_trainlist.txt` (seeded, disjoint, test IDs excluded). It supports `--freeze-flow`, AMP, grad clip 1.0, optional cosine LR, flip/temporal-reverse/optional-crop augmentation, and a batch-size probe that halves on OOM. Before training it records the epoch-0 (pretrained) val PSNR/SSIM. It saves `best.pt`/`last.pt` in the legacy format plus `train_ids`, `val_ids`, optimizer, scaler and history, and writes `config.json` and `history.json`. `--resume` was verified.
- Loss: L1 recon + 0.5·warp + 0.005·smooth (the reference ratio 204:102:1 without the perceptual term). With the flow net frozen, warp and smooth carry no gradient, so the effective objective is L1 reconstruction. **No perceptual loss**, because it needs VGG weights (a download).
- Verified on a fake Vimeo layout with a random full-width checkpoint: batch 6 at 256×448 with AMP fits in the 4 GB RTX 3050 Ti (frozen flow net).
- **Status: not run on real data**, because Vimeo-90K and the checkpoint are missing.

## Phase 6: evaluation
- `scripts/evaluate.py`: triplet (t = 0.5), septuplet (im1, im5 → im2..im4) or `--synthetic N`. Metrics are PSNR/SSIM on uint8 RGB/255, with `structural_similarity(channel_axis=-1, data_range=1.0)` and PSNR capped at 100 dB. It writes `summary.json` (merged per dataset, with N, std, runtime/frame, device and GPU) and `per_sample.csv`, and saves the 5 best and 5 worst comparisons by Super SloMo PSNR. It asserts that the official train and test lists are disjoint and that no evaluated triplet ID was in the fine-tune train/val subset; the guard was checked to fire.
- Run now: `--synthetic 30` with lk/farneback/average → `outputs/synthetic_eval/`. **Vimeo evaluation is pending data.**

## Open issues / needs from the user
1. `checkpoints/SuperSloMo.ckpt` (manual download from the avinashpaliwal/Super-SloMo README).
2. `data/vimeo_triplet/` (and optionally `data/vimeo_septuplet/`).
3. Real clips with motion/occlusion in `data/clips/` (videos or `frame0.png`/`frame1.png`[/`gt.png`] folders).
4. After (1): run `check_checkpoint.py --compare-warp-modes` on a real pair, inspect the image and pick the warp mode.
