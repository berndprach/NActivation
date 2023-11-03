
# N-Activation

Code for our paper 
[1-Lipschitz Neural Networks are more expressive with N-Activations](link). \
It includes [code](src/models/layers/activations/n_activation.py) for the proposed $\mathcal{N}$-activation.

## Requirements:
- Python 3.9
- PyTorch 1.12


## Usage:
Specify ```settings_nr``` (see [settings](settings.py) file) and provide it to the [training script](train.py): \
```python train.py 10```
