"""Notebook-friendly figures; flow is model-estimated, not ground-truth flow."""
from pathlib import Path
import numpy as np
import torch
from .preprocess import tensor_to_float01


def plot_history(history, path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), layout='constrained')
    for key, label in [('train_loss', 'train total'), ('train_recon_l1', 'train L1'), ('val_recon_l1', 'validation L1')]:
        points = [(h['epoch'], h[key]) for h in history if h.get(key) is not None]
        if points:
            axes[0].plot(*zip(*points), label=label)
    axes[0].set_title('Loss (train may use augmented crops)')
    axes[0].legend()
    for ax, key in zip(axes[1:], ['val_psnr', 'val_ssim']):
        ax.plot([h['epoch'] for h in history], [h[key] for h in history])
        ax.set_title(key)
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.3)
    if path:
        destination = Path(path)
        if destination.suffix.lower() != '.png':
            destination = destination / 'curves.png'
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


def _figure_directory(path):
    # Accept existing callers passing before.png; write before/*.png instead.
    path = Path(path)
    directory = path.with_suffix('') if path.suffix.lower() == '.png' else path
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def flow_rgb(flow, scale):
    import matplotlib.colors as colors
    xy = flow.detach().float().cpu().numpy().transpose(1, 2, 0)
    angle = (np.arctan2(xy[..., 1], xy[..., 0]) + np.pi) / (2 * np.pi)
    magnitude = np.linalg.norm(xy, axis=-1)
    return colors.hsv_to_rgb(np.stack([angle, np.ones_like(angle), np.clip(magnitude / scale, 0, 1)], -1))


@torch.no_grad()
def save_diagnostics(model, dataset, path=None, device='cpu', index=0):
    """Return three large comparison rows; optionally save each as a PNG."""
    import matplotlib.pyplot as plt
    from .preprocess import pad_to_multiple, crop_to
    model.eval()
    b = dataset[index]
    i0, i1 = [b[k].unsqueeze(0).to(device) for k in ('I0', 'I1')]
    p0, size = pad_to_multiple(i0)
    p1, _ = pad_to_multiple(i1)
    out = {k: crop_to(v, size) for k, v in model(p0, p1, 0.5).items()}
    rgb = lambda x: tensor_to_float01(x)[0]
    pred, gt = rgb(out['frame']), rgb(b['It'].unsqueeze(0))
    flows = [out[k][0] for k in ('F_01', 'F_10', 'F_t0_refined', 'F_t1_refined')]
    scale = max(float(torch.stack([f.square().sum(0).sqrt().max() for f in flows]).max()), 1e-6)
    panels = [(rgb(i0), 'Input t=0'), (gt, 'Ground truth t=0.5'), (pred, 'Prediction t=0.5'),
              (rgb(i1), 'Input t=1')]
    panels += [(flow_rgb(f, scale), title) for f, title in zip(flows, ['Estimated F01', 'Estimated F10', 'Refined Ft0', 'Refined Ft1'])]
    panels += [(np.abs(pred - gt).mean(-1), 'Mean absolute RGB error [0,1]'),
               (out['V_t0'][0, 0].cpu().numpy(), 'Visibility toward input 0'),
               ((rgb(i0) + rgb(i1)) / 2, 'Average baseline'),
               (np.abs((rgb(i0) + rgb(i1)) / 2 - gt).mean(-1), 'Average baseline error [0,1]')]
    rows = [('01_frames', panels[:4]),
            ('02_optical_flow', panels[4:8]),
            ('03_errors_visibility', panels[8:])]
    figures = {}
    for name, row in rows:
        fig, axes = plt.subplots(1, 4, figsize=(28, 7), layout='constrained')
        for ax, (img, title) in zip(axes, row):
            if img.ndim == 2:
                im = ax.imshow(img, vmin=0, vmax=1, cmap='inferno')
                fig.colorbar(im, ax=ax, fraction=.035, pad=.02, shrink=.75)
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=17, pad=12)
            ax.axis('off')
        caption = str(b['id'])
        if name == '02_optical_flow':
            caption += f' | hue=direction, brightness=magnitude; shared max={scale:.2f} px'
        fig.suptitle(caption, fontsize=18)
        figures[name] = fig
        if path:
            directory = _figure_directory(path)
            fig.savefig(directory / f'{name}.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
    return figures
