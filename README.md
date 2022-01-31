# ECGAN

ECGAN is a modular repository to train ML algorithms - and especially Generative Adversarial Networks (GANs) -
on electrocardiographic data, even though the setup is generally suitable for arbitrary time series.
While ECGAN also supports other tasks, such as classification using simple RNNs/CNNs, the focus lies on
the generation of data using GANs.
The GANs are trained on only the ''normal'' (for ECGs: healthy) class to learn the underlying representation of
this class. The resulting GAN can be used to generate new healthy data or detect anomalies by comparing
the synthetic data with the test data.
ECGAN follows modular setup allows easy modifications to the different parts of this project, allowing researchers to
simply  download and preprocess various ECG datasets, train a variety models to generate realistic time series and
apply a variety of anomaly detection mechanisms. ECGAN is built to be reproducible and easily expandable to allow fast
prototyping and foster comparability and open source implementations by reducing the time required to implement
models or preprocessing scenarios.

**Parts of the implementation are still in an experimental state.**

### Installation
#### Unix
Make sure to have installed python3.8 and git.

0. Clone the repository (using `git clone`)
1. Setup the virtual environment (e.g.`make env` which executes `virtualenv --python=/usr/bin/python3.8 .venv`
    and activate the environment `source .venv/bin/activate`)
2. Run `make setup_dev` in root if you want to develop. Run `make setup` if you only want to execute experiments.
3. During Step 2, ecgan and all requirements will be installed.

#### Windows

If you want to use ecgan on Windows, you cannot use `make` without additional tools. You can either use chocolatey
or similar tools to install `make` and use it. Otherwise, you can manually recreate the steps:
Install virtualenv (`pip install virtualenv`), create the environment `virtualenv --python=python3.8 .venv` and
activate the env (`.venv\Scripts\activate`) and install the packages (`pip install -U -r requirements.txt`)

### How to use

The package comes with a convenient CLI and configuration over YAML-Files.
By calling `ecgan-init -d DATASET -m MODULE -o FILE_NAME PROJECT NAME` a configuration file
is generated. This can be manually updated and contains most relevant hyperparameters.
After applying changes to the config file if desired you can start the preprocessing of
the chosen DATASET using `ecgan-preprocess` before training the model by invoking `ecgan-train`.
To perform anomaly detection, the `-a model_reference` flag is used (see [docs](https://emundo.github.io/ecgan-docs) for more
information).

### Data
We support several datasets will be used for evaluation: MIT-BIH (small amount of patients, long recordings, often
used), the PTB ECG dataset as well as the Shaoxing dataset (many patients, smaller recordings, published recently).

Some of the datasets require setting up kaggle credentials or downloading it manually (see [docs](https://emundo.github.io/ecgan-docs))

## About

### Project

ECGAN is an initiative of eMundo GmbH and the LMU Munich research group data mining in medicine supported by the
[Bavarian Research Foundation](https://forschungsstiftung.de/Welcome.html). The purpose of this initiative is to
investigate anomaly detection in time series data, focusing on electrocardiogram data. Do not hesitate to reach out if
you have any questions.
