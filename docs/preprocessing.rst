Preprocessing
=============

The framework supports preprocessing of various popular ECG datasets by default. After initializing a config file
you can simply invoke the preprocessing using :code:`ecgan-preprocess`.
The preprocessing follows the following procedure:

1. Download the dataset and relevant information required for labeling the data. Each preprocessor downloads the data
   from a different source - you might need to follow dataset specific instructions. Especially, you might need to
   configure the kaggle API or download the data manually from the respective repositories. The downloaded files are
   stored in :code:`<data_path>/<dataset_name>/raw`.

2. The downloaded data can then be further preprocessed. The exact preprocessing depends on the dataset since some
   datasets are already preprocessed. In general, we support cleansing, imputation,
   resampling to a target sequence length/frequence and windowing.

3. The preprocessed data is saved to :code:`<data_path>/<dataset_name>/processed` which always contains two pkl files
   with the data and labels. Both are saved as numpy arrays. The data is saved as a three dimensional Tensor of shape
   :code:`(num_samples, seq_len, num_channels)` for the data and as a one dimensional Tensor for the labels. The labels
   currently have to be integers which encode the different classes. Tasks which utilize a notion of anomalies
   assume that the 0 class is the normal class and that every other class is an abnormal class.


Some operations which would usually count as preprocessing, especially data transformations or channel selections.
These operations are performed in memory to avoid unnecessary persistent storage. To reproduce the preprocessing of
and given dataset you need to make sure that both configurations, the stored data from :code:`ecgan-preprocess` and
the configured in-memory changes, are correct.

For a list of supported datasets, see :ref:`Datasets`.
