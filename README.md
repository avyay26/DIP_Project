# Super SloMo: an optical-flow based method for frame interpolation using deep learning

Link to the paper: [arXiv:1712.00080](https://arxiv.org/abs/1712.00080) (PDF in this folder).

Reproduction and validation of **Super SloMo** (Jiang et al., CVPR 2018). The package contains the following:

- a full-width model that is layer-for-layer compatible with the pretrained `SuperSloMo.ckpt` from `avinashpaliwal/Super-SloMo`,
- Lucas–Kanade and Farnebäck baselines,
- direct multi-frame and streaming video interpolation,
- an optional small fine-tune and a held-out evaluation.

See `NOTES.md` for the working log and `REPORT.md` for the report.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate                  # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt         # includes torch 2.6.0 (CUDA 12.6); edit the index line for CPU-only
python -c "import torch, cv2, skimage; print(torch.__version__, torch.cuda.is_available(), cv2.__version__, skimage.__version__)"
```

## External assets (placed manually; the code never downloads anything)

| Asset | Default path | Flag |
| --- | --- | --- |
| `SuperSloMo.ckpt` (keys `state_dictFC`, `state_dictAT`) | `checkpoints/SuperSloMo.ckpt` | `--checkpoint` |
| Vimeo-90K triplets | `data/vimeo_triplet/` | `--vimeo-triplet` |
| Vimeo-90K septuplets (optional) | `data/vimeo_septuplet/` | `--vimeo-septuplet` |
| Test clips | `data/clips/` | `--input` / `--inputs` |

## Main commands

```bash
pytest -q                                                    # unit tests
SUPERSLOMO_CKPT=checkpoints/SuperSloMo.ckpt pytest -q tests/test_checkpoint.py   # + real checkpoint test

python scripts/smoke_test.py --device cpu --out outputs/smoke   # synthetic demo, no data needed (<1 min)

python scripts/check_checkpoint.py --checkpoint checkpoints/SuperSloMo.ckpt \
    --frame0 data/clips/pair/frame0.png --frame1 data/clips/pair/frame1.png \
    --out outputs/pretrained/check --compare-warp-modes

python scripts/interpolate.py --method superslomo --checkpoint checkpoints/SuperSloMo.ckpt \
    --frame0 a.png --frame1 b.png --times 0.25 0.5 0.75 --out outputs/pretrained/pair
python scripts/interpolate.py --method superslomo --checkpoint checkpoints/SuperSloMo.ckpt \
    --input data/clips/clip.mp4 --factor 4 --out outputs/pretrained/clip_4x.mp4 --batch-size 1
python scripts/interpolate.py --method lk        --input data/clips/clip.mp4 --factor 4 --out outputs/lk/clip_4x.mp4
python scripts/interpolate.py --method farneback --input data/clips/clip.mp4 --factor 4 --out outputs/farneback/clip_4x.mp4

python scripts/compare.py --inputs data/clips --methods lk farneback superslomo \
    --checkpoint checkpoints/SuperSloMo.ckpt --out outputs/eval

python scripts/finetune.py --checkpoint checkpoints/SuperSloMo.ckpt --vimeo-triplet data/vimeo_triplet \
    --num-train 700 --num-val 100 --epochs 2 --batch-size 6 --lr 1e-5 --freeze-flow --seed 0 --amp \
    --out runs/finetune

python scripts/evaluate.py --vimeo-triplet data/vimeo_triplet --split test --max-samples 500 \
    --methods lk farneback pretrained finetuned \
    --checkpoint checkpoints/SuperSloMo.ckpt --finetuned runs/finetune/best.pt --out outputs/eval
python scripts/evaluate.py --vimeo-septuplet data/vimeo_septuplet --split test --max-samples 300 \
    --methods lk farneback pretrained finetuned --times 0.25 0.5 0.75 \
    --checkpoint checkpoints/SuperSloMo.ckpt --finetuned runs/finetune/best.pt --out outputs/eval
```

Without any data, `evaluate.py --synthetic 30 --methods lk farneback average` and `compare.py --synthetic 3` run on rendered scenes. They are pipeline checks only.

## Layout

```text
superslomo/  model.py checkpoint.py data.py preprocess.py baselines.py metrics.py video.py inference.py viz.py
scripts/     smoke_test.py check_checkpoint.py interpolate.py compare.py finetune.py evaluate.py
tests/       test_model.py test_checkpoint.py test_data.py test_inference.py test_baselines.py test_metrics.py test_smoke.py
```

## Midpoint training on the collected dataset

Open `midpoint_training.ipynb` on the GPU server. It runs both experiments and displays saved loss/metric curves, prediction versus ground truth, error maps, visibility and estimated optical flow. Nothing downloads automatically and training is not run on the local machine for the full model.

Your `dataset/train` contains 500 folders, each with `im1.png`, `im2.png`, `im3.png`. Inputs are im1 and im3; **only im2 at t=0.5 is supervised and predicted**. The model is time-conditioned: requesting three times produces three outputs, while `model.interpolate(I0, I1, [0.5])` returns one midpoint. No output-layer change is needed. `scripts/interpolate.py --times 0.5` also predicts just the midpoint.

```bash
# Fine-tune refinement network; omit --freeze-flow to fine-tune both networks.
python scripts/finetune.py --dataset dataset/train --checkpoint checkpoints/SuperSloMo.ckpt --num-train 400 --num-val 50 --num-test 50 --epochs 20 --batch-size 4 --crop 256 256 --lr 1e-5 --freeze-flow --amp --cosine --out runs/midpoint_finetune

# Train BOTH reference U-Nets from random initialization; no checkpoint required.
python scripts/finetune.py --dataset dataset/train --scratch --split-file runs/midpoint_finetune/split.json --epochs 100 --batch-size 4 --crop 256 256 --lr 1e-4 --amp --cosine --out runs/midpoint_scratch
```

Crops must fit the source images. Training triplets in a batch must have equal sizes; cropping ensures this. Validation and test use full images, one at a time with padding removed after prediction. Reduce crop or batch size to save memory. Use distinct output directories for independent experiments. Resume with `--resume <run>/last.pt` and the same dataset/split, crop, freeze mode, learning-rate schedule and total epoch budget. Increase the epoch budget for a new experiment rather than expecting exact schedule continuation.

Each run writes `best.pt` (best validation PSNR, including the initial model), `last.pt` (resumable training state), `history.json`, `config.json`, `split.json`, `test_metrics.json`, and `curves/`, `before/`, `after/`, `test/` (individual PNG figures). The test set is evaluated using the selected best checkpoint. The JSON split can also be supplied manually with disjoint `train`, `val`, `test` lists of relative folder IDs. If samples share a source video, group by source video when constructing this split to avoid leakage. The generated random split cannot infer source-video relationships from folder names.

### How the repository works

- `superslomo/model.py`: reference `UNet` building block and complete `SuperSloMo`. The flow U-Net maps six endpoint RGB channels to four bidirectional flow channels. The refinement U-Net maps 20 channels to four flow corrections plus one visibility logit. Warping and visibility fusion synthesize the requested frame. `SuperSloMo(width="full")` already constructs randomly initialized reference networks; loading a checkpoint replaces those weights. The tiny width is only for smoke tests and is incompatible with the supplied full-width weights.
- `checkpoint.py`: strict legacy `state_dictFC` / `state_dictAT` loading and compatible saving. `preprocess.py`: RGB normalization and padding. `data.py`: Vimeo, collected-folder and synthetic loaders.
- `scripts/finetune.py`: training, validation-based checkpoint selection, held-out testing for collected data, optional frozen flow, AMP and resume. Loss is L1 reconstruction plus endpoint/intermediate warping and flow smoothness; no VGG perceptual loss is included. With flow frozen, only reconstruction provides parameter gradients.
- `inference.py`, `video.py`, `scripts/interpolate.py`: pair/video inference; factor 4 requests three in-between frames, whereas factor 2 requests one.
- `baselines.py`, `metrics.py`, `scripts/evaluate.py`, `scripts/compare.py`: classical flow baselines and comparisons. The existing evaluation CLI uses Vimeo layouts; collected-data testing is handled by the new training path and notebook.
- `viz.py`: comparison grids; `training_viz.py`: training curves and midpoint diagnostics. `training_pipeline.ipynb` is the original workflow; `midpoint_training.ipynb` is the collected-data workflow.

Scratch training usually needs more optimization than fine-tuning. The 20/100 epoch budgets above are starting experiments, not measured convergence claims. With only 400 training triplets, scratch training may overfit and may never match pretrained performance. Use validation curves to choose a training budget, and reserve test results for final comparisons. Epoch duration depends on server GPU, resolution, batch size and storage; `history.json` records measured epoch seconds, so estimate total time after a few representative epochs. Estimated flow has no ground-truth flow accuracy score because the dataset supplies RGB midpoint targets only.

### Separate figures and testing a trained checkpoint

New runs save individual PNG figures under `curves/`, `before/`, `after/`, and `test/` instead of combined grids. Diagnostic folders contain `I0.png`, `I1.png`, `gt.png`, `prediction.png`, four flow figures, error maps, visibility and the average baseline. `save_diagnostics` and `plot_history` return dictionaries of named figures for notebook display. Existing combined PNGs from earlier runs are left in place.

Evaluate `best.pt` on the saved held-out split without retraining (also regenerates separate figures):

```bash
python scripts/evaluate_midpoint.py --checkpoint runs/midpoint_finetune/best.pt --dataset dataset/train --split-file runs/midpoint_finetune/split.json --device cuda --visualize 3 --out outputs/midpoint_test
```

Scores are written to `metrics.json`; figures go into `sample_000/`, etc. Test targets are im2; only im1 and im3 enter the model. For scratch weights, change the checkpoint path to `runs/midpoint_scratch/best.pt`.

To predict from any new endpoint pair without ground truth:

```bash
python scripts/interpolate.py --method superslomo --checkpoint runs/midpoint_finetune/best.pt --frame0 path/to/im1.png --frame1 path/to/im3.png --times 0.5 --device cuda --out outputs/new_pair
```

The predicted image is `outputs/new_pair/superslomo_t0.500.png`. Ground truth is required to compute PSNR/SSIM, but is not needed for prediction.

Diagnostic figures now group each original row into its own large PNG: `01_frames.png` (I0, GT, prediction, I1), `02_optical_flow.png` (four flows), and `03_errors_visibility.png` (prediction error, visibility, average baseline, baseline error). Each row uses a 28-by-7-inch figure exported at 200 DPI. Training curves remain separate plots. Older individual PNG files remain in existing output folders; use a new output directory when regenerating to view only the new rows.

Training curves are saved together in `curves.png`: loss, validation PSNR, and validation SSIM in one three-panel figure. `plot_history` returns that figure. Diagnostic row images remain separate.
