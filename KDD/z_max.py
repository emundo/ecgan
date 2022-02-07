"""Reproduction of "Change of the largest z value in training data"."""
# Samples are exported from the corresponding Weights and Biases run.
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(
    'KDD/data/max_mean_z.csv',
    delimiter=';',
    skip_header=1,
    names=['x', 'BeatGANplus', 'BeatGAN', 'VAEGANplus', 'VAEGAN'],
)

plt.plot(data['x'], data['BeatGAN'], color="#1A85FF", label='BeatGAN', lw=0.8)
plt.plot(data['x'], data['BeatGANplus'], color="#1A85FF", ls='dashed', label='BeatGAN$_+$', lw=0.8)
plt.plot(data['x'], data['VAEGAN'], color="#D41159", label='$\\beta$-VAEGAN', lw=0.8)
plt.plot(data['x'], data['VAEGANplus'], color="#D41159", ls='dashed', label='$\\beta$-VAEGAN$_+$', lw=0.8)

plt.xlabel('Epoch', fontsize=10)
plt.ylabel('$z_{max}$')  # plt: 6.4x4.8inches -> .75
fig_width, fig_height = plt.gcf().get_size_inches()
plt.gcf().set_size_inches(3.5, 2)
plt.legend(fontsize='x-small')
# plt.show()
plt.savefig('KDD/z_max.pgf', dpi=400, bbox_inches='tight', pad_inches=0.0001)
