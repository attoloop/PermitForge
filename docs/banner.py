"""Draw the banner from one recorded episode: python docs/banner.py EPISODE.json docs/banner.jpg

A plan view of the plant: buildings, stacks at their submitted heights, the fence, and every receptor
within 500 m coloured by how much room the submitted design left under the hourly NO2 limit there.
Needs matplotlib (`uv run --with matplotlib python docs/banner.py ...`).
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

INK, PAPER, RULE, MUTED, SOFT = '#1B1E24', '#FFFFFF', '#D9DBE0', '#6B7078', '#EEF0F2'
SAGE = ['#D9E6E2', '#9ABAB1', '#5F8E84', '#33605A']  # light to dark: more room under the limit, then less


def main(episode_path: str, out: str) -> None:
    record = json.load(open(episode_path))
    geometry = record['geometry']
    design = record['steps'][-1]['args']['design']
    margins = record['steps'][-1]['receptor_margins']['NO2_1h']
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0.03, 0.09, 0.94, 0.84])
    ax.set_facecolor(PAPER)
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-560, 560)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(Rectangle((-250, -250), 500, 500, fill=False, edgecolor=RULE, linewidth=1.4, linestyle=(0, (5, 5))))
    for b in geometry['buildings']:
        ax.fill([c[0] for c in b['corners']], [c[1] for c in b['corners']], color=SOFT, edgecolor=INK, linewidth=1.2, zorder=2)
    shown = [(r, m) for r, m in zip(geometry['receptors'], margins) if max(abs(r[0]), abs(r[1])) <= 520]
    ordered = sorted(m for _, m in shown)
    cap = ordered[int(len(ordered) * .9)]  # the darkest tenth is the near-binding set
    cmap = LinearSegmentedColormap.from_list('sage', SAGE[::-1])
    ax.scatter([r[0] for r, _ in shown], [r[1] for r, _ in shown], s=85, c=[cmap(min(m, cap) / cap) for _, m in shown],
               edgecolors=PAPER, linewidths=.9, zorder=3)
    for s in geometry['stacks']:
        height = design[s['id']]['height_m']
        ax.scatter([s['x_m']], [s['y_m']], s=60 + 9 * height, facecolors=PAPER, edgecolors=INK, linewidths=2, alpha=.9, zorder=4)
    binding = next(st for st in record['final']['standards'] if st['id'] == 'NO2_1h')['binding_receptor']
    ax.scatter([binding['x_m']], [binding['y_m']], s=360, facecolors='none', edgecolors=INK, linewidths=1.8, zorder=5)
    ax.text(binding['x_m'] + 24, binding['y_m'] + 26, 'binding receptor', fontsize=11.5, color=INK, family='monospace', va='bottom')
    lines = [f"{s['id']}  {design[s['id']]['control'].split('_')[-1]:<8} {design[s['id']]['height_m']:>6g} m" for s in geometry['stacks']]
    ax.text(600, 330, 'submitted design\n' + '\n'.join(lines), fontsize=12, color=INK, family='monospace', va='top',
            linespacing=1.55, bbox=dict(facecolor=PAPER, edgecolor=RULE, boxstyle='square,pad=0.8'))
    fig.text(0.03, 0.965, f"{geometry['task_id']} · station {geometry['station']} · hourly NO₂ standard · submitted design",
             fontsize=12.5, color=MUTED, family='monospace', va='top')
    fig.text(0.03, 0.035, 'receptor colour: room left under the limit (dark = little) · circle size: stack height · dashed: the fence',
             fontsize=11.5, color=MUTED, family='monospace', va='bottom')
    fig.savefig(out, facecolor=PAPER, pil_kwargs={'quality': 88})


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
