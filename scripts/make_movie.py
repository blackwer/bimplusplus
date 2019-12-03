#!/usr/bin/env python3

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.animation import FFMpegWriter


def make_movie(filename):
    with h5py.File(filename, "r") as f:
        positions_t = np.array(f["positions_t"])
        theta_t = np.array(f["theta_t"])
        U_t = np.array(f["U_t"])
        alpha = np.array(f["alpha"])
        area_n = np.array(f["area_n"])[0]
        nrecord = np.array(f["nrecord"])[0]
        dt = np.array(f["dt"])[0]

    fbase = os.path.splitext(filename)[0]
    metadata = dict(title=fbase)
    writer = FFMpegWriter(fps=30, metadata=metadata)

    fig = plt.figure(figsize=[6.4, 6.4], dpi=100)
    with writer.saving(fig, fbase + ".mp4", 100):
        for i in range(0, positions_t.shape[0]):
            positions = positions_t[i, :, :]
            thetas = theta_t[i, :]
            U_n = U_t[i, :]

            normals = np.vstack([-np.sin(thetas), np.cos(thetas)]).T

            plt.plot(
                np.sqrt(area_n / np.pi) * np.hstack([np.cos(alpha), np.cos(alpha[0])]),
                np.sqrt(area_n / np.pi) * np.hstack([np.sin(alpha), np.sin(alpha[0])]),
                color="red",
            )
            plt.plot(
                np.hstack([positions[:, 0], positions[:, 0]]),
                np.hstack([positions[:, 1], positions[:, 1]]),
                color="blue",
            )
            plt.quiver(
                positions[:, 0],
                positions[:, 1],
                U_n * normals[:, 0],
                U_n * normals[:, 1],
                scale=2.5,
                color="blue",
                headwidth=1.5,
                headlength=2.5,
            )
            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])
            writer.grab_frame()

            plt.cla()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Input '.h5' file required")
    else:
        make_movie(sys.argv[1])
