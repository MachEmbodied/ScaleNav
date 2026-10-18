"""Plot ScaleNav body-frame displacements in the initial robot frame."""
import argparse
from pathlib import Path

import numpy as np


def integrate_actions(actions):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 3 or not len(actions):
        raise ValueError('Expected a nonempty (N, 3) trajectory')
    if not np.isfinite(actions).all():
        raise ValueError('Trajectory contains nonfinite values')
    path = np.zeros((len(actions) + 1, 3), dtype=np.float64)
    for index, (forward, right, yaw) in enumerate(actions):
        x, y, heading = path[index]
        c, s = np.cos(heading), np.sin(heading)
        path[index + 1] = (x + c * forward - s * right,
                           y + s * forward + c * right, heading + yaw)
    return path


def plot_trajectory(actions, output, image=None):
    path = integrate_actions(actions)
    output = Path(output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, 2 if image is not None else 1,
        figsize=(9, 3.3) if image is not None else (4, 4), squeeze=False,
        gridspec_kw={'width_ratios': [1.8, 1]} if image is not None else None,
    )
    try:
        if image is not None:
            axes[0, 0].imshow(plt.imread(image) if isinstance(image, (str, Path)) else image)
            axes[0, 0].set_title('Observation')
            axes[0, 0].axis('off')
        plot = axes[0, -1]
        plot.plot(path[:, 1], path[:, 0], '-o', color='#2563eb', markersize=2.5,
                  linewidth=1.8)
        plot.scatter(0, 0, color='#16a34a', s=28, zorder=3, label='Start')
        plot.scatter(path[-1, 1], path[-1, 0], color='#dc2626', s=28,
                     zorder=3, label='End')
        plot.quiver(path[-1, 1], path[-1, 0], np.sin(path[-1, 2]), np.cos(path[-1, 2]),
                    angles='xy', scale_units='inches', scale=6, color='#dc2626')
        distance = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()
        plot.set(xlabel='Right (m)', ylabel='Forward (m)')
        plot.set_title(f'Next {len(actions)} actions · path {distance:.2f} m', fontsize=10)
        span = max(3.0, 1.4 * float(np.ptp(path[:, 0])),
                   1.4 * float(np.ptp(path[:, 1])))
        center = (path[:, :2].min(axis=0) + path[:, :2].max(axis=0)) / 2
        plot.set_xlim(center[1] - span / 2, center[1] + span / 2)
        plot.set_ylim(center[0] - span / 2, center[0] + span / 2)
        plot.set_aspect('equal', adjustable='box')
        plot.tick_params(labelsize=8)
        plot.grid(alpha=.2)
        plot.legend(frameon=False, fontsize=8, loc='upper left', ncol=2,
                    columnspacing=.8, handletextpad=.3)
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
    finally:
        plt.close(figure)
    print(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/example/trajectory.png'))
    args = parser.parse_args(argv)
    plot_trajectory(np.load(args.trajectory, allow_pickle=False), args.output, args.image)


if __name__ == '__main__':
    main()
