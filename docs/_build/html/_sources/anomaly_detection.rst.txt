Anomaly Detection
=================
We supported various ways to detect anomalies (for ECG data: arrhythmias and morphological abnormalities) in ECGAN.
This includes anomaly detection via classification (using RNNs/CNNs) as well as using a generative approach
(see :ref:`AnoGAN`).

AnoGAN
~~~~~~
Schlegl et al. 2017 propose `AnoGAN (paper)<https://arxiv.org/abs/1703.05921>` to detect arrhythmias using GANs:
They train a GAN solely on the normal class and detect anomalies by comparing an arbitrarily similar sample which can be trained
using the generator with the real sample during inference. By measuring and weighting the difference in data space and
the discriminator score, a sample can be classified as real or synthetic or respectively normal and abnormal.
Some relevant choices are:

1. How the ''arbitrarily similar sample'' is retrieved. AnoGAN uses an expensive iterative optimization in latent space.
   Subsequent work (`ALAD (paper)<https://arxiv.org/abs/1812.02288>`) proposes learning an inverse mapping
   (e.g. using cycle consistency) during training of after training to significantly reduce the inference time.

2. How the difference in data space is measured: Usually, the L1/L2 distance is used. While this is not a good measure
   for time series data, it is currently sufficient for most ''simple'' application (e.g. beatwise segmented ECG data)

3. How the discriminator is used: We can use a target value but this value is rather instable and changes a lot during
   training. AnoGAN uses feature matching to reduce this problem.

4. How the scores are weighted: AnoGAN and most subsequent work uses a fixed and linear weighting of the scores.
   However, this might be far from the best choice and a grid search or a non-linear combination can be used.
