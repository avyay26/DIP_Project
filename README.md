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
