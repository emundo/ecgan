"""Reproduction of "Influence of the spectral normalization using discriminator and reconstruction loss (SVM)"."""
# Samples are exported from the corresponding Weights and Biases run.
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(
    'KDD/data/spectral_lambda_fscore.csv', delimiter=';', skip_header=1, names=['x', 'spectral', 'no_spectral']
)

plt.plot(data['x'], data['spectral'], color="#1A85FF", lw=0.8, label='weight norm.')
plt.plot(data['x'], data['no_spectral'], color="#D41159", lw=0.8, label='input norm.')
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('$F_1$')  # plt: 6.4x4.8inches -> .75
plt.gcf().set_size_inches(3.5, 2)
plt.legend(loc='lower right', fontsize='x-small')
# plt.show()
plt.savefig('KDD/spectral_lambda_svm.pgf', dpi=400, bbox_inches='tight', pad_inches=0.0001)
