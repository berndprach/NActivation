from src import models


def get_model(model_name: str,
              activation_name: str,
              conv_name: str,
              activation_kwargs: dict = None,
              **kwargs):
    model_cls = getattr(models, model_name)

    if activation_kwargs is None:
        activation_kwargs = {}

    original_activation_cls = getattr(models.layers, activation_name)

    def activation_cls(in_channels):
        if "NActivation" in activation_name:
            return original_activation_cls(in_channels, **activation_kwargs)
        else:
            return original_activation_cls(**activation_kwargs)

    if conv_name == "AOLConv2d":
        return model_cls(
            activation_cls=activation_cls,
            conv_cls=models.layers.AOLConv2d,
            first_conv_cls=models.layers.AOLConv2dOrthogonal,
            head_conv_cls=models.layers.AOLConv2dOrthogonal,
            **kwargs,
        )

    conv_cls = getattr(models.layers, conv_name)
    return model_cls(
        activation_cls=activation_cls,
        conv_cls=conv_cls,
        **kwargs,
    )
