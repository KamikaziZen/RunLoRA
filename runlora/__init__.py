"""The RunLoRA framework contains efficient implementation of LoRA
that significantly improves the speed of neural network training
and fine-tuning using low-rank adapters.

The proposed implementation optimizes the computation of LoRA operations
based on dimensions of corresponding linear layer, layer input dimensions
and lora rank by choosing best forward and backward computation graph
based on FLOPs and time estimations,
resulting in faster training without sacrificing accuracy.
"""
from .runlora import *

__author__ = "Aleksandr Mikhalev and Daria Cherniuk"
__credits__ = ["Aleksandr Mikhalev", "Daria Cherniuk", "Ivan Oseledets"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Aleksandr Mikhalev and Daria Cherniuk"
__email__ = 'al.mikhalev@skoltech.ru, daria.cherniuk@skoltech.ru, kamikazizen@gmail.com'
__status__ = "Development"
