Supported Functionality
=================================
This page serves as a concise overview of datasets and preprocessing, training and
evaluation methods currently supported by ECGAN.

.. note::
    Not all preprocessing methods supported by the framework are used for
    every dataset and not every metric is in use for every method.
    If you feel like we are missing crucial preprocessing or evaluation steps
    please get in touch!

Supported Datasets
---------------------------------
The dataset can best be set during initialization using the :code:`-b` flag followed by the identifier below.
We currently support the following datasets out of the box:


+------------------------+------------------------+----------------------------------------------------------+------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Dataset                | Identifier             | Dataset                                                  | Retriever                                                                    | Preprocessor                                                                |
+========================+========================+==========================================================+==============================================================================+=============================================================================+
| MITBIH :sup:`1`        | :code:`mitbih`         | :class:`ecgan.utils.datasets.MitbihDataset`              | :class:`ecgan.preprocessing.data_retrieval.MitbihDataRetriever`              | Not supported                                                               |
+                        +------------------------+----------------------------------------------------------+------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
+                        | :code:`mitbih_beats`   | :class:`ecgan.utils.datasets.MitbihExtractedBeatsDataset`| :class:`ecgan.preprocessing.data_retrieval.MitbihExtractedBeatsDataRetriever`| :class:`ecgan.preprocessing.preprocessor.MitbihExtractedBeatsPreprocessor`  |
+                        +------------------------+----------------------------------------------------------+------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
+                        | :code:`mitbih_beatgan` | :class:`ecgan.utils.datasets.MitbihBeatganDataset`       | :class:`ecgan.preprocessing.data_retrieval.MitbihBeatganDataRetriever`       | :class:`ecgan.preprocessing.preprocessor.MitbihBeatganPreprocessor`         |
+------------------------+------------------------+----------------------------------------------------------+------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
+ Shaoxing :sup:`2`      | :code:`shaoxing`       | :class:`ecgan.utils.datasets.ShaoxingDataset`            | :class:`ecgan.preprocessing.data_retrieval.ShaoxingDataRetriever`            | :class:`ecgan.preprocessing.preprocessor.ShaoxingPreprocessor`              |
+------------------------+------------------------+----------------------------------------------------------+------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
+ PTB :sup:`3`           | :code:`ptb`            | :class:`ecgan.utils.datasets.PTBExtractedBeatsDataset`   | :class:`ecgan.preprocessing.data_retrieval.PtbExtractedBeatsDataRetriever`   | :class:`ecgan.preprocessing.preprocessor.PtbExtractedBeatsPreprocessor`     |
+------------------------+------------------------+----------------------------------------------------------+------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
+ Synthetic sine :sup:`4`| :code:`sine`           | :class:`ecgan.utils.datasets.SineDataset`                | :class:`ecgan.preprocessing.data_retrieval.SineDataRetriever`                | Not supported                                                               |
+------------------------+------------------------+----------------------------------------------------------+------------------------------------------------------------------------------+-----------------------------------------------------------------------------+

| :sup:`1` a. Paper: `Moody and Mark 2001 <https://ieeexplore.ieee.org/abstract/document/932724>`_, more information
  on the `PhysioNet MITBIH Website <https://physionet.org/content/mitdb/1.0.0/>`_. Data source: unofficial
  `kaggle mirror <https://www.kaggle.com/mondejar/mitbih-database/>`_. The data remains unchanged but is saved as csv.
  This dataset needs manual preprocessing by the user, we only support the download.
| :sup:`1` b. Paper: `Kachuee et al. 2018 <https://arxiv.org/abs/1805.00794>`_. Data source:
  `official Kaggle repository <https://www.kaggle.com/shayanfazeli/heartbeat>`_. Classes are according to the AAMI classification, each beat is classified individually.
| :sup:`1` c. Paper: `Zhou et al. 2019 <https://www.ijcai.org/Proceedings/2019/0616.pdf>`_. Data source:
  `official Dropbox mirror <https://www.dropbox.com/sh/b17k2pb83obbrkn/
   AABF9mUNVdaYwce9fnwXsg1ta/ano0?dl=0&subfolder_nav_tracking=1>`_. Data is centered, standardized, classes are according to AAMI, each beat is classified individually.
| :sup:`2` Paper: `Zheng et al. 2020 <https://pubmed.ncbi.nlm.nih.gov/32051412/>`_. Data source:
      `official figshare mirror <https://figshare.com/collections/ChapmanECG/4560497/2>`_.
| :sup:`3` Original Paper: `Bousseljot et al. 1995 <https://www.deepdyve.com/lp/de-gruyter/nutzung-der-ekg-signaldatenbank-cardiodat-der-ptb-ber-das-internet-uemKpjIFzM>`_
        Paper of data source: `Kachuee et al. 2018 <https://arxiv.org/abs/1805.00794>`_.
        Data source: `official Kaggle repository <https://www.kaggle.com/shayanfazeli/heartbeat>`_.
| :sup:`4` Artificial sine wave data, description can be found here: :class:`ecgan.preprocessing.data_retrieval.SineDataRetriever`.


Additional public ECG datasets which are currently not supported include the recent PTB-XL from `Wagner et al. 2020 <https://www.nature.com/articles/s41597-020-0495-6>`_
and ECG 5000. More information on ECG datasets can be found on the `PhysioNet Database <https://physionet.org/about/database/>`_.

Not all datasets are suitable for all tasks, the quality of the Shaoxing dataset is for example not necessarily high enough for reliable data generation.
We hope to add more information and in-depth evaluations for the supported datasets in the future.

Supported Preprocessing
---------------------------------
The preprocessing is not set as a flag during initialization, but can be changed after initialization, before starting :code:`ecgan-preprocess`.
Most importantly, you can select the sequence length (:code:`TARGET_SEQUENCE_LENGTH`) and downsample or upsample the data.
This is very important since the sequence length can be very important for the performance of a model
and sometimes needs to be downsampled for computational requirements or upsampled to be a good fit
for an existing architecture with a specific sequence length in mind.
For downsampling we suggest using `LTTB <https://pypi.org/project/lttb/>`_ downsampling (:code:`DOWNSAMPLE_ALGO: lttb`) to retain the structure of the time series.
For upsampling we support the torch interpolation (:code:`DOWNSAMPLE_ALGO: interpolate`) which uses linear interpolation between the values.

Supported Training
------------------
During training data is split into n Folds (:code:`CROSS_VAL_FOLDS` in the configuration file).
Many parameters such the desired (amount of) channels (either a list for the selected indices or an integer m
to take the first m channels), flags to indicate if data shall be masked for binary classification
(:code:`BINARY_LABELS`), transformations such as various normalizations or the Fourier transform (:code:`TRANSFORMATION`)
can be set. It is further possible to train solely on healthy data which can be useful for some tasks such
as various generative tasks.
Other parameters which are of relevance for training many models (currently focusing deep learning models)
can be set freely such as the amount of epochs or the batch size.

Supported Models
----------------
Can be selected using the :code:`-m` flag.

1. Data generation/synthesis:

    a. Traditional GAN based models (especially DCGAN/RGAN) with a variety of :ref:`Loss functions`. Usage :code:`-m dcgan`, :code:`-m rgan`, :code:`-m rdcgan`.

    b. Autoencoder based GANs: using a (variational) autoencoder. Usage :code:`-m aegan`, :code:`-m vaegan`.

2. Data classification:

    a. Simple RNN for (multiclass as well as single class) classification. Usage :code:`-m rnn`.

    b. Simple CNN for (multiclass as well as single class) classification. Usage :code:`-m cnn`.

Supported trackers
------------------
Training and evaluation can be tracked using :ref:`Tracker` which can be set using the :code:`TRACKER`
parameter in the experiment config. Currently supported options are local tracking (:code:`TRACKER: local`)
and Weights and Biases (:code:`TRACKER: wb`). While local tracking is set as a default to not force people to
sign up for another service to test ecgan, we strongly recommend Weights and Biases!

Further downstream tasks
------------------------
The most important downstream tasks is the anomaly detection, see :ref:`Anomaly Detection` for detailed information.
Supported techniques currently focus on pretrained models and include both, classification as well as generative models.

Additionally, inverse mappings can be trained for selected deep generative models. See :ref:`Inverse Mapping` for more information.