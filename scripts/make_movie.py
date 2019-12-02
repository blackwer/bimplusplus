#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.animation import FFMpegWriter

with h5py.File("test.h5", "r") as f:
    positions_t = np.array(f["positions_t"])
    theta_t = np.array(f["theta_t"])
    U_t = np.array(f["U_t"])
    alpha = np.array(f["alpha"])
    area_n = np.array(f["area_n"])[0]
    n_record = np.array(f["n_record"])[0]
    dt = np.array(f["dt"])[0]

fig = plt.figure()
metadata = dict(title="Movie Test", artist="Me", comment="Testing. Duh.")
writer = FFMpegWriter(fps=30, metadata=metadata)

with writer.saving(fig, "test.mp4", positions_t.shape[0]):
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
