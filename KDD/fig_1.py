"""Reproduction of "Heatmap of normal and abnormal real and reconstructed sample"."""
# Samples are randomly selected from test set.
# The .pkl contains a nested list of [real_samples, fake_samples].
import pickle

import matplotlib.pyplot as plt
import numpy as np

with open("KDD/data/random_samples_fig_1.pkl", "rb") as f:
    samples = pickle.load(f)

real_samples = np.array(samples[0])
fake_samples = np.array(samples[1])

fig, axes = plt.subplots(2, 2, sharex=True, gridspec_kw={'height_ratios': [6, 1]})
data_lined_healthy = real_samples[2]
data_dashed_healthy = fake_samples[2]
heatmap_data_0 = np.abs(data_lined_healthy - data_dashed_healthy)
data_lined_unhealthy = real_samples[5]
data_dashed_unhealthy = fake_samples[5]
heatmap_data_1 = np.abs(data_lined_unhealthy - data_dashed_unhealthy)
max_val = max(heatmap_data_0) if max(heatmap_data_0) > max(heatmap_data_1) else max(heatmap_data_1)
COLOR_LINED = "blue"
COLOR_DASHED = "red"

len_ = len(data_lined_healthy) if len(data_lined_healthy) > len(data_dashed_unhealthy) else len(data_dashed_unhealthy)
x_axis = np.arange(len_)
axes[0][0].plot(x_axis, data_lined_healthy, color=COLOR_LINED, label='real')
axes[0][0].plot(x_axis, data_dashed_healthy, color=COLOR_DASHED, linestyle="--", label='synth')
axes[0][0].legend()
axes[1][0].imshow([heatmap_data_0], aspect="auto", cmap="plasma", vmin=0, vmax=max_val)
axes[0][1].plot(x_axis, data_lined_unhealthy, color=COLOR_LINED, label='real')
axes[0][1].plot(x_axis, data_dashed_unhealthy, color=COLOR_DASHED, linestyle="--", label='synth')
axes[0][1].legend()
heatmap = axes[1][1].imshow([heatmap_data_1], aspect="auto", cmap="plasma", vmin=0, vmax=max_val)
axes[1][0].get_yaxis().set_visible(False)
axes[1][1].get_yaxis().set_visible(False)
axes[1][0].get_xaxis().set_visible(False)
axes[1][1].get_xaxis().set_visible(False)
axes[0][0].set_xlabel("Normal Class")
axes[0][1].set_xlabel("Abnormal Class")
plt.tight_layout()
fig.colorbar(heatmap, ax=axes.ravel().tolist(), label='Absolute Error')

plt.gcf().set_size_inches(3.5, 2)
# plt.show()

plt.savefig('KDD/Figure_1_label.png')
