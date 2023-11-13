
# N-Activation

Code for our paper 
[1-Lipschitz Neural Networks are more expressive with N-Activations](https://arxiv.org/abs/2311.06103). \
It includes [code](src/models/layers/activations/n_activation.py) for the proposed $\mathcal{N}$-activation.

## Requirements:
- Python 3.9
- PyTorch 1.12


## Usage:
Use the ```settings_nr``` (see below or [settings](settings.py) file) to specify the setting and run:

    python train.py 10

The last digit of the settings number determins the activation, 0 means **MaxMin** and 1 for the $\mathcal{N}$-**activation**.
The setting number also determins the dataset and method for making the convolutions 1-Lipschitz, see below.

| Settings Nr | Dataset       | Method   |
|:------------|:--------------|:---------|
| 10-19       | CIFAR-10      | AOL      |
| 20-29       | CIFAR-10      | CPL      |
| 30-39       | CIFAR-10      | SOC      |
| 40-49       | CIFAR-100     | AOL      |
| 50-59       | CIFAR-100     | CPL      |
| 60-69       | CIFAR-100     | SOC      |
| 70-79       | Tiny ImageNet | AOL      |
| 80-89       | Tiny ImageNet | CPL      |
| 90-99       | Tiny ImageNet | SOC      |

