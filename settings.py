
from src.default_settings import DefaultSettings

# Datasets:
CIFAR10 = "CIFAR10"
CIFAR100 = "CIFAR100"
TINY_IMAGENET = "TinyImageNet"

DATASET_LABEL_DIMENSIONS = {
    CIFAR10: 10,
    CIFAR100: 100,
    TINY_IMAGENET: 200,
}

DATASET_BASE_WIDTHS = {
    CIFAR10: 64,
    CIFAR100: 64,
    TINY_IMAGENET: 32,
}

DATASET_NROF_BLOCKS = {
    CIFAR10: 5,
    CIFAR100: 5,
    TINY_IMAGENET: 6,
}

# Layers:
AOL = "AOLConv2d"
CPL = "CPLConv2d"
SOC = "SOCConv2d"

# Learning rates:
LOG10_LRS = {
    AOL: -1.6,
    CPL: -0.4,
    SOC: -1.,
}


BASE_SETTINGS = {
    1: (CIFAR10, AOL),
    2: (CIFAR10, CPL),
    3: (CIFAR10, SOC),
    4: (CIFAR100, AOL),
    5: (CIFAR100, CPL),
    6: (CIFAR100, SOC),
    7: (TINY_IMAGENET, AOL),
    8: (TINY_IMAGENET, CPL),
    9: (TINY_IMAGENET, SOC),
}


def get_settings(settings_nr: int):
    setting = DefaultSettings()

    base_settings_nr = settings_nr // 10
    activation_settings_nr = settings_nr % 10

    dataset_name, layer_name = BASE_SETTINGS[base_settings_nr]

    setting.log_folder = f"FINAL_{layer_name}-{dataset_name}-N{settings_nr}"
    set_architecture_settings(setting, dataset_name)
    setting.conv_name = layer_name
    set_activation_settings(setting, activation_settings_nr)

    setting.use_test_set = True

    setting.log_lr = LOG10_LRS[layer_name]
    setting.momentum = 0.9
    setting.weight_decay = 0.

    return setting


def set_architecture_settings(setting: DefaultSettings, dataset_name: str):
    base_width = DATASET_BASE_WIDTHS[dataset_name]
    label_dim = DATASET_LABEL_DIMENSIONS[dataset_name]
    nrof_blocks = DATASET_NROF_BLOCKS[dataset_name]

    setting.dataset_name = dataset_name
    setting.model_kwargs["nrof_classes"] = label_dim
    setting.model_kwargs["base_width"] = base_width
    setting.model_kwargs["nrof_blocks"] = nrof_blocks


def set_activation_settings(setting, activation_settings_nr):

    # MaxMin (for main experiments):
    if activation_settings_nr == 0:
        setting.activation_name = "MaxMin"
        setting.model_kwargs["activation_kwargs"] = {}

    # NActivation (for main experiments):
    elif activation_settings_nr == 1:
        setting.activation_name = "NActivation"
        lr_factor = 10 ** (-2.)
        setting.model_kwargs["activation_kwargs"] = {
            "theta_init_values": (-100., 0., 0., 0.),
            "lr_factor": lr_factor,
        }

    # NActivation, "zero initialization" (for ablation study):
    elif activation_settings_nr == 2:
        setting.activation_name = "NActivation"
        lr_factor = 10 ** (-2.)
        setting.model_kwargs["activation_kwargs"] = {
            "theta_init_values": (0., 0.),
            "lr_factor": lr_factor,
        }

    # NActivation, "random initialization" (for ablation study):
    elif activation_settings_nr == 3:
        setting.activation_name = "RandomLogUniformNActivation"
        lr_factor = 10 ** (-2.)
        setting.model_kwargs["activation_kwargs"] = {
            "lr_factor": lr_factor,
        }

