Anomaly Detection Submodule
===========================
This submodule contains relevant functionality to perform anomaly detection on a test dataset.
The data split, configuration to correctly preprocess and reconstruct the dataset and the trained model are
saved during training. The components are reconstructed in the manager file (:func:`ecgan.manager.detect`). The
anomaly manager class (:class:`ecgan.anomaly_detection.anomaly_manager`) performs the evaluation on the test data set.

.. toctree::
   :maxdepth: 2
   :glob:

   *
   Detectors <detector/index>
