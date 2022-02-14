About
=============
The ECGAN framework was initially developed as part of a research project between
`eMundo GmbH <https://www.e-mundo.de/en>`_ and the research group
`Data Mining in Medicine <https://dmm.dbs.ifi.lmu.de/cms/>`_ of LMU Munich.

ECGAN is generously supported and partially funded by the `Bavarian Research Foundation
<https://forschungsstiftung.de/Welcome.html>`_.
The focus of the research project was to investigate and improve the state of the art
generation of ECG data using Generative Adversarial Networks. However, the framework
has since expanded and supports various additional tasks.

Goal
----
ECGAN aims to provide several things:

1. | First and foremost, reproducibility and comparability is central to the idea
   | of ECGAN. Algorithms should either be deterministic or means to analyse their
   | performance across several runs should be possible.
2. | Generalizable foundation. While the documentation, metrics and datasets focus on
   | ECG data, it should remain easy to evaluate any time series problem (see :ref:`Datasets`).
3. | Focus on ECG data. We aim to provide information to people in the intersection of
   | data science and cardiology by providing information on available datasets and
   | their use. Over time, we aim to provide real scenarios - given boundaries and data -
   | and discuss how various preprocessing steps can influence the resulting data/models.
   | This will require significant help from the community - especially to keep up with
   | SOTA methods - which is why we appreciate any help or hints regarding novel/interesting
   | research.

Contributing
------------

Get started with the development
--------------------------------
If you want to help develop the framework we recommend to install a variety of development
tools, especially mypy and pylint. The tools can be installed using :code:`make setup_dev`
and the code can be checked for lint/type stub/formatting errors using :code:`make check`.

We strongly encourage discussions (either in the issues or via mail at emubot@e-mundo.de)
regarding the possible errors (and even better: fixes) of the current pipeline and addition
or improvements of datasets, preprocessing steps or training/anomaly detection modules.

Improving Baselines
-------------------
Current baselines are sometimes rather simplistic. Improvements can be very simple (e.g.
exchanging the input normalization) and we aim to achieve high performances on given
datasets with given preprocessing and a specific task (e.g. single-beat classification,
multi-beat classification or beat generation).
If you are able to improve an existing baseline, please use either 5 or 10-fold
crossvalidation (i.e.set the :code:`trainer.CROSS_VAL_FOLDS` parameter to 5 or 10). Please report the
resulting task metrics (see :ref:`Data Synthesis`). We will
additionally run the code independently to verify the result before merging.
