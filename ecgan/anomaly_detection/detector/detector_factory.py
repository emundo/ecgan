"""Factory to return AnomalyDetector objects."""
from ecgan.anomaly_detection.detector.base_detector import AnomalyDetector
from ecgan.anomaly_detection.detector.classification_detector import ArgmaxClassifierDetector
from ecgan.anomaly_detection.detector.reconstruction_detector import GANAnomalyDetector, GANInverseAnomalyDetector
from ecgan.evaluation.tracker import BaseTracker
from ecgan.modules.base import BaseModule
from ecgan.modules.classifiers.nn_classifier import NNClassifier
from ecgan.modules.generative.base import BaseGANModule
from ecgan.utils.custom_types import AnomalyDetectionStrategies, ReconstructionType


class AnomalyDetectorFactory:
    """Meta module for creating correct anomaly detectors."""

    @staticmethod
    def choose_class(module: AnomalyDetectionStrategies):
        """Choose the correct class based on the provided module name."""
        anomaly_detectors = {
            AnomalyDetectionStrategies.ANOGAN: GANAnomalyDetector,
            AnomalyDetectionStrategies.ARGMAX: ArgmaxClassifierDetector,
            AnomalyDetectionStrategies.INVERSE_MAPPING: GANInverseAnomalyDetector,
        }
        try:
            return anomaly_detectors[module]
        except KeyError as err:
            raise AttributeError('Argument {0} is not set correctly.'.format(module)) from err

    def __call__(self, detector: str, module: BaseModule, tracker: BaseTracker) -> AnomalyDetector:
        """Return implemented AD module when a BaseModule is created."""
        if detector == AnomalyDetectionStrategies.ANOGAN.value:
            if not isinstance(module, BaseGANModule):
                raise TypeError(
                    'BaseGANModule is expected for AnoGAN detection, current model type: {0}'.format(type(module))
                )

            return GANAnomalyDetector(
                module=module,
                reconstructor=ReconstructionType.INTERPOLATE,
                tracker=tracker,
            )
        if detector == AnomalyDetectionStrategies.INVERSE_MAPPING.value:
            if not isinstance(module, BaseGANModule):
                raise TypeError(
                    'BaseGANModule is expected for inverse mapping, current module type: {0}'.format(type(module))
                )

            return GANInverseAnomalyDetector(
                module=module,
                reconstructor=ReconstructionType.INVERSE_MAPPING,
                tracker=tracker,
            )
        if detector == AnomalyDetectionStrategies.ARGMAX.value:
            if not isinstance(module, NNClassifier):
                raise TypeError('NNClassifier is expected, current type: {0}'.format(type(module)))

            return ArgmaxClassifierDetector(module=module, tracker=tracker)
        raise AttributeError('Argument {0} is not set correctly.'.format(detector))
