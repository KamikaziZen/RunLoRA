# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import List, Any
import math
import bitsandbytes as bnb


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


class RunLoRALinear(nn.Module, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_operator,
        quantization_config,
        lora_r: int,
        lora_alpha: int,
        device=None,
        dtype=None,
        lora_dropout: float = 0.,
        weight=None,
        bias=None,
        keep_original=False,
    ):
        assert lora_r > 0, 'LoRA rank must be positive'

        super().__init__()
        LoRALayer.__init__(self, lora_r=lora_r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout)

        self.in_features = in_features
        self.out_features = out_features
        self.keep_original = keep_original

        if weight is None:
            weight_data = torch.empty(in_features, out_features, dtype=dtype, device=device)
        else:
            weight_data = weight.data.detach().t().contiguous()

        if not quantization_config:
            # runlora weight is transponent to usual torch.Linear weights
            # weight_data = weight_data.t()
            self.weight = nn.Parameter(weight_data,
                                       # Freezing the pre-trained weight matrix
                                       requires_grad=False)
        elif quantization_config.load_in_4bit:
            self.weight = bnb.nn.Params4bit(
                weight_data,
                requires_grad=False,
                quant_state = weight.quant_state,
                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                quant_type=quantization_config.bnb_4bit_quant_type,
            )
        elif quantization_config.load_in_8bit:
            self.weight = bnb.nn.Int8Params(
                weight_data,
                requires_grad=False,
            )
        else:
            raise ValueError("Wrong format of quantization config")

        if bias is not None:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype),
                                     requires_grad=False)
            self.bias.data = bias.data.detach().contiguous()
        else:
            self.register_parameter('bias', None)

        self.lora_U = nn.Parameter(
            torch.empty((in_features, lora_r), dtype=dtype, device=device))
        self.lora_V = nn.Parameter(
            torch.empty((lora_r, out_features), dtype=dtype, device=device))
        self.scaling = self.lora_alpha / self.lora_r

        # Initializing weights
        self.reset_parameters()

        # Setting forward and backward paths
        self.lora_operator = lora_operator

    def reset_parameters(self):
        # initialized a tensor of size (m, n) can have substantially larger
        # norm than initialized a tensor of size (n, m)
        # thus, transposition is added to W and U init
        # to ensure approximately the same norm
        # with default peft.lora initialization
        if not self.keep_original:
            nn.init.kaiming_uniform_(self.weight.T, a=math.sqrt(5))
            # TODO: get rid of fan_in, fan_out?
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.lora_U.T, a=math.sqrt(5))
        nn.init.zeros_(self.lora_V)

    # def train(self, mode: bool = True):
    #     nn.Linear.train(self, mode)
    #     # set lora dropout here to train?

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        # to make consistent with 
        # x = x.to(lora_A.weight.dtype)
        # result = result.to(torch_result_dtype) (peft implementation)
        # https://github.com/huggingface/peft/blob/8e979fc73248ccb4c5b5a99c415f3e14a37daae6/src/peft/tuners/lora/layer.py#L514C13-L514C51
        if x.dtype in (torch.bfloat16, torch.half):
            result_dtype = x.dtype
        else:
            result_dtype = self.weight.dtype
        # x = x.to(self.lora_U.dtype)
        # TODO: this is not correct if p_drop > 0
        # lora_dropout should be applied to x only for A and B
        # result = self.lora_operator.apply(
        #     self.lora_dropout(x), self.weight, self.lora_U * self.scaling, self.lora_V, self.bias)
        result = self.lora_operator.apply(
            x, self.weight, self.lora_U * self.scaling, self.lora_V, self.bias)
        # return result
        return result.to(result_dtype)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, ' \
               f'out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, ' \
               f'lora_r={self.lora_r}, ' \
               f'lora_alpha={self.lora_alpha}, ' \
               f'forward={self.lora_operator.forward.__name__}, ' \
               f'backward={self.lora_operator.backward.__name__}, ' \
               f'weight_dtype={self.weight.dtype}'

    @classmethod
    def from_linear(cls, module, lora_operator, **kwargs):
        self = cls(module.in_features,
                   module.out_features,
                   lora_operator,
                   weight=module.weight,
                   bias=module.bias,
                   keep_original=True,
                   **kwargs)

        return self


class RunLoRAModel(nn.Module):
    def __init__(self,
                 model: PreTrainedModel,
                 run_lora_mapping,
                 target_modules: List[str],
                 lora_r: int,
                 lora_alpha: int,
                 lora_dtype,
                 lora_dropout: float = 0.):

        super().__init__()

        self.base_model = model
        self.config = self.base_model.config
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device = model.device

        self.forward = self.base_model.forward

        for module_name, module in self.base_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(trgt in module_name for trgt in target_modules):
                continue

            # new_module = RunLoRALinear(
            #     module.in_features,
            #     module.out_features,
            #     run_lora_mapping[module_name],
            #     weight=None,
            #     bias=module.bias,
            #     lora_r=self.lora_r,
            #     lora_alpha=self.lora_alpha,
            #     lora_dropout=self.lora_dropout,
            #     quantization_config=self.config.quantization_config
            # )

            new_module = RunLoRALinear.from_linear(
                module,
                run_lora_mapping[module_name],
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                quantization_config=self.config.quantization_config if hasattr(self.config, 'quantization_config') else None,
                device=self.device,
                dtype=lora_dtype
            )

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.base_model.get_submodule(parent_name)
        return parent

    def prepare_for_finetuning(self, modules_to_save=None):
        for name, param in self.base_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            elif modules_to_save and any(m in name for m in modules_to_save):
                param.requires_grad = True
            else:
                param.requires_grad = False
