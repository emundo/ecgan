References
==========
This page is meant to be a centralized source of information for ECGAN.
This includes

#. a list of abbreviations used in the code and documentation,
#. distinctions between terms used,
#. references to papers, repositories and other information from external sources.

The references are usually also included as source in the implementation.

Related Work
------------

Data
~~~~~
#. MITBIH (`Moody and Mark, 2001 <http://ecg.mit.edu/george/publications/mitdb-embs-2001.pdf>`_, `MITBIH Website <https://physionet.org/content/mitdb/1.0.0/>`_):

    a. Original MITBIH (no additional preprocessing, binary classification from kaggle): `Original MITBIH Dataset <https://www.kaggle.com/mondejar/mitbih-database>`_

    b. MITBIH per-beat, classes according to AAMI (non-binary by default): `Kachuee et al. 2018 <https://arxiv.org/abs/1805.00794>`_, `Preprocessed MITBIH Dataset <https://www.kaggle.com/shayanfazeli/heartbeat>`_

    c. MITBIH BeatGAN: per-beat, centered, standardized, classes according to AAMI (non-binary), removed some patients: `Zhou et al. 2019 <https://www.ijcai.org/Proceedings/2019/0616.pdf>`_, `Dataset <https://www.dropbox.com/sh/b17k2pb83obbrkn/AADzJigiIrottyTOyvAEU1LOa?dl=0>`_, `Repository <https://github.com/Vniex/BeatGAN>`_

#. Shaoxing (`Zheng et al. 2020 <https://www.nature.com/articles/s41597-020-0386-x>`_, `Shaoxing Dataset <https://figshare.com/collections/ChapmanECG/4560497/2>`_)
#. Not supported but also of relevance: PTB (`Bousseljot et al. 1995 <https://www.degruyter.com/document/doi/10.1515/bmte.1995.40.s1.317/html>`_) and PTB-XL (`Wagner et al. 2020 <https://www.nature.com/articles/s41597-020-0495-6>`_).

Related Modules
~~~~~~~~~~~~~~~
#. Generative Models

    #. DCGAN: Paper: `Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Radford et al. 2015 <https://arxiv.org/abs/1511.06434>`_, Repository:  `DCGAN GitHub <https://github.com/Newmu/dcgan_code>`_

    #. RGAN: Paper: `Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs, Esteban et al. 2017 <https://arxiv.org/pdf/1706.02633.pdf>`_, Repository: `RGAN GitHub <https://github.com/ratschlab/RGAN>`_

    #. BeatGAN: Paper: `BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series, Zhou et al. 2019 <https://www.ijcai.org/proceedings/2019/0616.pdf>`_, Repository: `BeatGAN Github <https://github.com/Vniex/BeatGAN>`_

    #. VaeGAN: BeatGAN as well as "Survival-oriented embeddings with application to CT scans of colorectal carcinoma patients with liver metastases" by Tobias Weber, 2021.

The modules have received significant improvements, often inspired by more recent developments. This
especially includes:

    #. The Wasserstein distance (including gradient penalties) as objective function. Papers: `Wasserstein GAN, Arjovsky et al. 2017 <https://arxiv.org/abs/1701.07875>`_ and `Improved Training of Wasserstein GANs, Gulrajani et al. 2017 (WGAN-GP) <https://arxiv.org/abs/1704.00028>`_

    #. Spectral norm as weight normalization (usually instead of input normalization in respective layers): `Spectral Normalization for Generative Adversarial Networks, Miyato et al. 2018 <https://arxiv.org/abs/1802.05957>`_

#. Anomaly detection

    #. AnoGAN: Optimization of latent variables towards similar samples, Paper: `Schlegl et al. 2017 <https://arxiv.org/pdf/1703.05921.pdf>`_ .

