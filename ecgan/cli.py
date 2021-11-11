"""Definition of ecgan CLI commands."""
import ecgan


def run_init():
    """Initialize ecgan configuration."""
    ecgan.init()


def run_preprocessing():
    """Run preprocessing process."""
    ecgan.preprocess()


def run_training():
    """Run training process."""
    ecgan.train()


def run_detection():
    """Run anomaly detection."""
    ecgan.detect()


def run_inverse():
    """Run training of the inverse mapping."""
    ecgan.inverse()
