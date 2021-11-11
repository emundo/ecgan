Datasets
=================================

As ECGAN aims to support a large variety of ECG datasets and tries to be compatible with
other tasks related to time series, creating and using new datasets is a key feature.

Role in the ECGAN Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~
Datasets are the input to any model used in this framework and we expect them to be implemented in
a consistent way. The dataset is given as an argument during initialization of the config and used
during all steps - preprocessing, training and evaluation. The default values can impact the
suggested parameter choices and are preconfigured depending on the dataset. This means that generating a new
config file for a new dataset is usually better than just manually changing the name of the dataset in the config.

Adding new Datasets
~~~~~~~~~~~~~~~~~~~
To add a new datasets please follow these steps:

#. Think of an identifier which describes your dataset (e.g.:code:`my_ecg`).
#. Add a descriptive class for your dataset which inherits from :code:`ecgan.utils.datasets.Dataset` (e.g.
   :code:`MyEcgDataset`) and add it to the :code:`ecgan.utils.datasets.DatasetFactory`.
#. Add retrieval class to download the data and store it in a data directory with
   the prefix :code:`DATASET_NAME/raw` (e.g. :code:`data/my_ecg/raw`. The stored
   information can be split into an arbitrary amount of files at this point. Add the
   class to the :code:`ecgan.preprocessing.data_retrieval.DataRetrieverFactory`.
#. Add preprocessing class for your dataset inheriting from
   :code:`ecgan.preprocessing.preprocessor.Preprocessor`. The output is saved into
   :code:`DATASET_NAME/processed` (e.g. :code:`data/my_ecg/processed`). At this point
   the data has to conform to the dataset format: it has to be saved to
   the :code:`data.pkl` file as a 3D Tensor of shape :code:`(num_samples, seq_len, num_channels)`.
   Add the class to the :code:`ecgan.preprocessing.preprocessor.PreprocessorFactory` class.


Datasets supported by the framework can be found at :ref:`Supported Datasets`.

.. note::
    While the :class:`ecgan.utils.datasets.Dataset` class is used to describe arbitrary
    datasets, ECGAN further implements a :class:`ecgan.training.datasets.BaseDataset`.
    This is a class inheriting from the PyTorch :code:`dataset` class used to iterate
    through datasets. Keep in mind that these two are different.

.. automodule:: ecgan.utils.datasets
    :members:
    :show-inheritance: