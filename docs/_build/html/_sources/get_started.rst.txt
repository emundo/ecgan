Get Started
=================================
Setup and first steps
---------------------
To get started with the ECGAN framework, make sure to have `Python 3.8
<https://www.python.org/downloads/release/python-380/>`_ and `Pip
<https://pip.pypa.io/en/stable/installing/>`_ installed.
It is recommended to activate a virtual environment before installing the dependencies.
If you are on Unix (slight differences for Windows):

1. Clone the repository (e.g. :code:`git clone https://github.com/emundo/ecgan`) and go into the repository (:code:`cd ecgan`).
2. Install the dependencies (:code:`pip install -r requirements.txt`)
3. Generate a config file for preprocessing and training. Change the configuration file if desired.
   Example: :code:`ecgan-init -m cnn entity_name project_name run_name` (more information: :ref:`Config`) to generate a
   config file for a CNN classifier.
4. Use :code:`ecgan-preprocess` to download and preprocess the data as defined in the config.
5. Train your model using :code:`ecgan-train`.

If you want to use your model for an additional task (e.g. anomaly detection or create an inverse mapping for a trained
GAN), you can simply use the reference to a model saved during training to perform that task.

Example:
Given you have trained the above CNN classification model, you might want to detect anomalies based on
their maximum activation in the output layer. You might train multiple folds and save your model various times.
The best performing model during validation was model version 12. Using
:code:`ecgan-detect -a argmax -i entity_name/project_name/run_name:v12`
you can set the detection strategy to the argmax detector (which uses the maximum value for a
class as a prediction). The above command generates a anomaly detection config file which can
then be executed using :code:`ecgan-detect`.

Setting up tracking (W&B)
-----------------------------
By default, the data is saved locally inside the repository.
Since other tracking tools, such as MLFlow or Weights and Biases, often offer significant benefits
(such as improved visualization, more statistics and the possibility to easily view the results on
different machines), we offer an abstract tracking class which can be implemented for such platforms.
During development we have used Weights and Biases (W&B). Using the W&B adapter is simple:

1. Set up a weights and biases account. Afterwards you will have an entity name (e.g. your username or team name).
2. Create a config file (see :ref:`Setup and first steps` - make sure to enter the correct entity name!
3. Set the tracker inside the config to :code:`wb` and
4. Start your run!

Setting up kaggle [Optional]
----------------------------
To automatically download the data, some datasets require access to the kaggle API.
To set up kaggle, follow `these instructions <https://github.com/Kaggle/kaggle-api>`_.
While this is the easiest way to obtain the data, you can also directly download the
data from the Kaggle website (links are provided as documentation in the respective code
as well as in the :ref:`References`) and extract it to the folder where the raw data is
downloaded to (:code:`DATA_ROOT/DATASET_NAME/raw/`, e.g.:code:`data/mitbih_beats/raw/`).

Setting up S3/BOTO
------------------
Saving your models in Weights and Biases or locally can quickly add up and take a significant amount of space on your
disc. To avoid this, you can set the flag :code:`S3_CHECKPOINT_UPLOAD` to :code:`true` to only upload the data to S3.
To do this, we use `Boto3 <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html>`_.
Follow their setup to configure the AWS SDK before changing the flag.
