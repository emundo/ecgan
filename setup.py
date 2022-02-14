"""ECGAN setup configuration."""
from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ecgan',
    entry_points={
        'console_scripts': [
            'ecgan-init = ecgan.cli:run_init',
            'ecgan-preprocess = ecgan.cli:run_preprocessing',
            'ecgan-train = ecgan.cli:run_training',
            'ecgan-detect = ecgan.cli:run_detection',
            'ecgan-inverse = ecgan.cli:run_inverse',
        ]
    },
    version='0.0.2',
    description='Library to train and evaluate ML architectures to detect anomalies in time series, especially ECG.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/emundo/ecgan',
    author='Fiete Lüer, Maxim Dolgich, Tobias Weber',
    author_email='emubot@e-mundo.de',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[],
    python_requires='>=3.8, <4',
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
