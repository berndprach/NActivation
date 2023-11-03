
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict


@dataclass
class DefaultSettings:
    # Data:
    dataset_name: str = "CIFAR10"
    data_dir: str = "data"
    batch_size: int = 256
    use_test_set: bool = False

    # Model:
    model_name: str = "SimplifiedConvNet"
    conv_name: str = "AOLConv2d"
    activation_name: str = "MaxMin"
    model_kwargs: Dict = field(default_factory=lambda: {})

    # Loss
    loss_name: str = "OffsetCrossEntropyFromScores"
    loss_kwargs: Dict = field(default_factory=lambda: {
        "offset": 2 * 2**0.5 * 36 / 255,
        "temperature": 1/4,
    })

    # Optimization:
    log_lr: float = -3.
    weight_decay: float = 0.
    momentum: float = 0.9
    nrof_epochs: int = 1000

    # Logging:
    log_folder: str = "logs"

    _is_initialized: bool = False

    def __post_init__(self):
        self._is_initialized = True

    def __setattr__(self, name, value):
        """ Prevents adding new attributes to the class (e.g. typos). """
        if self._is_initialized and not hasattr(self, name):
            raise AttributeError(
                f"{type(self).__name__} instance has no attribute {name}")
        super().__setattr__(name, value)

    @property
    def as_dict(self):
        return asdict(self)


class AOLSettings(DefaultSettings):
    pass


class CPLSettings(DefaultSettings):
    conv_name: str = "CPLConv2d"


class SOCSettings(DefaultSettings):
    conv_name: str = "SOCConv2d"

