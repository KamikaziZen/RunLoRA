import torch
import torch.nn as nn
from transformers import PreTrainedModel
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


class LightLoRALinear(nn.Module, LoRALayer):
    # TODO: dropout
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_operator,
        bias: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float = 0.,
        **kwargs
    ):
        assert lora_r > 0, 'LoRA rank must be positive'

        super().__init__()
        LoRALayer.__init__(self, lora_r=lora_r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout)

        self.in_features = in_features
        self.out_features = out_features
        # transponent to usual torch.Linear weights
        self.weight = nn.Parameter(torch.empty((in_features, out_features)), **kwargs)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **kwargs))
        else:
            self.register_parameter('bias', None)

        self.lora_U = nn.Parameter(self.weight.new_zeros((in_features, lora_r)))
        self.lora_V = nn.Parameter(self.weight.new_zeros((lora_r, out_features)))
        self.scaling = self.lora_alpha / self.lora_r

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        # Initializing weights
        self.reset_parameters()

        # Setting forward and backward paths
        self.lora_operator = lora_operator

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.lora_U, a=math.sqrt(5))
        nn.init.zeros_(self.lora_V)

    # def train(self, mode: bool = True):
    #     nn.Linear.train(self, mode)
    #     # set lora dropout here to train?

    def forward(self, x: torch.Tensor):
        # TODO: scaling
        return self.lora_operator.apply(
            x, self.weight, self.lora_U, self.lora_V, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, ' \
               f'out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, ' \
               f'lora_r={self.lora_r}, ' \
               f'lora_alpha={self.lora_alpha}, ' \
               f'forward={self.lora_operator.forward.__name__}, ' \
               f'backward={self.lora_operator.backward.__name__}'


class LightLoRAModel(nn.Module):
    def __init__(self,
                 model: PreTrainedModel,
                 light_lora_mapping,
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
                light_lora_mapping[module_name],
                bias=True if module.bias else False,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.base_model.get_submodule(parent_name)
        return parent

    def prepare_for_finetuning(self):
        for name, param in self.base_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
