.. ECGAN documentation master file, created by
   sphinx-quickstart on Thu Jun 17 11:30:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the ECGAN framework!
=================================
The ECGAN framework provides a pipeline to preprocess, train and evaluate PyTorch models
trained on time series data. The focus lies on anomaly detection in ECG data: this
is highlighted by datasets that are supported out-of-the-box. Generally, the aim is to
offer tools to automatically preprocess the data, to provide a unified training and evaluation
pipeline and toanalyse data, such as spectral analysis (FFT) and embeddings (UMAP/t-SNE).

The current focus of implemented models lies on ECG generation and its use for the detection of rhythmic or morphologic
abnormalities in time series, but several other methods are already supported to serve as baseline detection algorithms.
The main goal is to offer **reproducible** implementations of modern algorithms. This results in a rather flexible
framework with simple ways to quickly develop new :ref:`training` or :ref:`anomaly detection` methods.
We aim to add several classical models for time series analysis (e.g. autoregressive models) in the near future. In the
long term we will try to offer more information and tools for various aspects of ECG data processing - from filtering
the raw signal to augmenting datasets.

Please get in touch if you have any suggestions for the framework or find missing or wrong information/implementation
details. We aim to provide reproducible state-of-the-art methods, so please feel free to improve our existing models
or add new models (see :ref:`contributing`) for more information.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   Get started <get_started>
   Framework Structure <structure>
   Supported Methods and Datasets <supported>
   Preprocessing <preprocessing>
   Data Synthesis <synthesis>
   Anomaly Detection <anomaly_detection>
   Python API <api/index>
   References <references>
   About <about>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
