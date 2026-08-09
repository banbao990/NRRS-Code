import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

plt.rcParams['font.family'] = 'Times New Roman'


def bin(d, filename):
    num_bins = d.shape[0]
    fig = plt.figure(figsize=(150 / 96, 200 / 96))
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)

    for i in range(num_bins):
        ax.bar(i, d[i], width=1.0, color='#ADD8E6', edgecolor="#1B58D1", alpha=1.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 8)

    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("-|>", size=1.5)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)

    # draw a horizontal line at the mean value
    mean_val = np.mean(d)
    ax.axhline(mean_val, color='#FF69B4', linestyle='--', linewidth=2, zorder=10)  # pink dashed line

    # plt.show()
    fig.savefig(filename, dpi=96, bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":
    v1 = 2
    v2 = 5
    v3 = 8
    num = 6
    dis = 1
    gen = False

    ds = [
        [7.78, 7.17, 7.27, 6.45, 6.17, 5.28],
        [3.71, 4.62, 3.88, 4.27, 3.69, 3.30],
        [7.23, 6.93, 6.98, 6.57, 6.43, 5.98],
    ]
    ds = np.array(ds)

    if gen:
        ds[0] = np.random.uniform(v2, v3, num)
        ds[1] = np.random.uniform(v1, v2, num)
        # ds[2] = np.random.uniform(v2 + dis, v3 - dis, num)
        mean = np.mean(ds[0])
        ds[2] = np.copy(ds[0])
        ds[2] = ds[2] - (ds[2] - mean) * 0.5

        print("[{}],".format(",".join(["{:.2f}".format(x) for x in ds[0]])))
        print("[{}],".format(",".join(["{:.2f}".format(x) for x in ds[1]])))
        print("[{}],".format(",".join(["{:.2f}".format(x) for x in ds[2]])))

    bin(ds[0], "images/loss_ori.svg")
    bin(ds[1], "images/loss_min.svg")
    bin(ds[2], "images/loss_avg.svg")
