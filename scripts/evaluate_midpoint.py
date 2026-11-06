"""Evaluate best.pt on saved held-out triplets, without retraining."""
import argparse
import json
from pathlib import Path
import _common
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
from superslomo.checkpoint import build_model_from_checkpoint
from superslomo.data import FolderTriplet
from superslomo.training_viz import save_diagnostics
from finetune import validate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset', default='dataset/train')
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='outputs/midpoint_test')
    ap.add_argument('--visualize', type=int, default=3)
    args = ap.parse_args()
    split = json.loads(Path(args.split_file).read_text())
    ids = split[args.split]
    if not ids:
        ap.error('Selected split is empty')
    if set(ids) & set(split.get('train', [])):
        ap.error('Evaluation IDs overlap training IDs')
    model, _ = build_model_from_checkpoint(args.checkpoint, args.device)
    ds = FolderTriplet(args.dataset, ids)
    metrics = validate(model, DataLoader(ds, batch_size=1), args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    for index in range(min(max(args.visualize, 0), len(ds))):
        save_diagnostics(model, ds, out / f'sample_{index:03d}', args.device, index)
    print(json.dumps(metrics, indent=2))
    print(f'Results saved to {out}')


if __name__ == '__main__':
    main()
