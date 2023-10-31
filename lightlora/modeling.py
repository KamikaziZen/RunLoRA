import torch
import torch.nn as nn
from transformers import PreTrainedModel
from lightlora import light_lora_collection
from typing import List
import math


class LoRALayer():
    def __init__(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class LightLoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        path_f: int,
        path_b: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float = 0.,
        **kwargs
    ):
        assert lora_r > 0, 'LoRA rank must be positive'

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, lora_r=lora_r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout)

        self.U = nn.Parameter(self.weight.new_zeros((lora_r, in_features)))
        self.V = nn.Parameter(self.weight.new_zeros((out_features, lora_r)))
        self.scaling = self.lora_alpha / self.lora_r

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        # Initializing weights
        self.reset_parameters()

        # Setting forward and backward paths
        self.light_lora_func = light_lora_collection[path_f, path_b].apply

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'U'):
            nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
            nn.init.zeros_(self.V)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        # set lora dropout here to train?

    def forward(self, x: torch.Tensor):
        return self.light_lora_func(x, self.weight.t(),
                                    self.U, self.V, self.bias)


class LighLoRAModel(nn.Module):
    def __init__(self,
                 model: PreTrainedModel,
                 path_f: int,
                 path_b: int,
                 target_modules: List[str],
                 lora_r: int,
                 lora_alpha: int,
                 lora_dropout: float = 0.):

        super().__init__()

        self.base_model = model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.forward = self.base_model.forward

        for module_name, module in self.base_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(trgt in module_name for trgt in target_modules):
                continue

            new_module = LightLoRALinear(
                module.in_features,
                module.out_features,
                path_f=path_f,
                path_b=path_b,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias=module.bias is not None,
            )

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.base_model.get_submodule(parent_name)
        return parent
