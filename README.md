# Super SloMo

Frame interpolation with Super SloMo, Lucas-Kanade, and Farneback.

Install dependencies with `pip install -r requirements.txt`. The pretrained weights are at `checkpoints/SuperSloMo.ckpt`.

## Training

Each sample in `dataset/train/<sample>/` contains `im1.png`, `im2.png`, and `im3.png`. The model uses the first and third frames to predict the middle frame.

Fine-tune the pretrained model:

```bash
python scripts/finetune.py --dataset dataset/train --checkpoint checkpoints/SuperSloMo.ckpt --num-train 400 --num-val 50 --num-test 50 --epochs 20 --batch-size 4 --crop 256 256 --lr 1e-5 --freeze-flow --amp --cosine --out runs/midpoint_finetune
```

Train from scratch using the same split:

```bash
python scripts/finetune.py --dataset dataset/train --scratch --split-file runs/midpoint_finetune/split.json --epochs 100 --batch-size 4 --crop 256 256 --lr 1e-4 --amp --cosine --out runs/midpoint_scratch
```

## Inference

Predict a middle frame:

```bash
python scripts/interpolate.py --method superslomo --checkpoint runs/midpoint_finetune/best.pt --frame0 frame0.png --frame1 frame1.png --times 0.5 --out outputs/pair
```

Interpolate a video:

```bash
python scripts/interpolate.py --method superslomo --checkpoint runs/midpoint_finetune/best.pt --input clip.mp4 --factor 4 --out outputs/slow_motion.mp4
```

For classical methods, use `--method lk` or `--method farneback`; no checkpoint is needed.

## Implementations
Dataset preparation in `dataset_fetch.ipynb`.
Checkpoint finetuning in `midpoint_training.ipynb` and model training from scratch in `scratch_training.ipynb`.
 

## Layout

```text
superslomo/               Model, datasets, preprocessing, baselines, and inference
scripts/                  Training, evaluation, and interpolation commands
midpoint_training.ipynb    Training notebook
checkpoints/              Pretrained weights
runs/                     Training checkpoints and logs
outputs/                  Predictions and evaluation results
```
