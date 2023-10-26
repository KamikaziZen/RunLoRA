import torch
from lightlora import *
from copy import deepcopy
from time import time
import torch.utils.benchmark as benchmark
from argparse import ArgumentParser
import pandas as pd
from torch.cuda.amp import custom_fwd, custom_bwd

parser = ArgumentParser(prog="Parameters for LightLora")
parser.add_argument("--n_batch", type=int, default=1)
parser.add_argument("--n_seq", type=int, default=4096)
parser.add_argument("--n_layer", type=int, default=1)
parser.add_argument("--n_in", type=int, default=1024)
parser.add_argument("--n_out", type=int, default=1024)
parser.add_argument("--n_rank", type=int, default=32)
parser.add_argument("--dtype", default="float")
parser.add_argument('-o', "--out", type=str, default='out')

args = parser.parse_args()
rows = []

class MLPLinear(torch.nn.Module):
    def __init__(self, n_layer, n_in, n_out):
        super().__init__()
        self.n_layer = n_layer
        self.linear1 = torch.nn.ModuleList([torch.nn.Linear(n_in, n_out, \
                bias=False) for i in range(n_layer)])
        self.linear2 = torch.nn.ModuleList([torch.nn.Linear(n_out, n_in, \
                bias=False) for i in range(n_layer)])

    def forward(self, x):
        y = x
        for i in range(self.n_layer):
            y = self.linear1[i](y)
            y = self.linear2[i](y)
        return y

class LoRA(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank, bias=True, \
            device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.U = torch.nn.Parameter(torch.randn(in_features, rank, \
                device=device, dtype=dtype))
        self.V = torch.nn.Parameter(torch.randn(rank, out_features,\
                device=device, dtype=dtype))
    
    def forward(self, x):
        y = super().forward(x) + (x@self.U)@self.V
        return y

class MLPLoRA(torch.nn.Module):
    def __init__(self, n_layer, n_in, n_out, n_rank):
        super().__init__()
        self.n_layer = n_layer
        self.lora1 = torch.nn.ModuleList([LoRA(n_in, n_out, n_rank, \
                bias=False) for i in range(n_layer)])
        self.lora2 = torch.nn.ModuleList([LoRA(n_out, n_in, n_rank, \
                bias=False) for i in range(n_layer)])

    def forward(self, x):
        y = x
        for i in range(self.n_layer):
            y = self.lora1[i](y)
            y = self.lora2[i](y)
        return y

class MLPLightLoRA(torch.nn.Module):
    def __init__(self, n_layer, n_in, n_out, n_rank):
        super().__init__()
        self.n_layer = n_layer
        self.lora1 = torch.nn.ModuleList([LightLoRA(n_in, n_out, n_rank, \
                bias=False) for i in range(n_layer)])
        self.lora2 = torch.nn.ModuleList([LightLoRA(n_out, n_in, n_rank, \
                bias=False) for i in range(n_layer)])

    def forward(self, x):
        y = x
        for i in range(self.n_layer):
            y = self.lora1[i](y)
            y = self.lora2[i](y)
        return y
    
    def flops(self, x):
        nflops = 0
        y = x
        with torch.no_grad():
            for i in range(self.n_layer):
                nflops += self.lora1[i].flops(y)
                y = self.lora1[i](y)
                nflops += self.lora2[i].flops(y)
                y = self.lora2[i](y)
        del y
        return nflops


def mytimeit(info, nflops):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    b = benchmark.Timer(stmt="mlp(x).sum().backward()", \
            globals={'mlp': mlp, 'x': x})
    measure_warmup = b.blocked_autorange(min_run_time=1.0)
    measure = b.blocked_autorange(min_run_time=1.0)
    print("Evaluating \"{}\"".format(info))
    print("Mean time: {} us".format(measure.mean * 1000000))
    print("GFlops: {}".format(nflops*1e-9))
    print("GFlops/s: {}".format(nflops*1e-9 / measure.mean))
    print("Max mem: {} MB".format(torch.cuda.max_memory_allocated()/2**20))
    return {'mean_time': measure.mean * 1000000,
            'Gflops': nflops * 1e-9, 
            'Gflops/s': nflops * 1e-9 / measure.mean,
            'max_mem_MB': torch.cuda.max_memory_allocated() / 2**20}

device = torch.device("cuda")
if args.dtype == "float" or args.dtype == "fp32":
    dtype = torch.float
elif args.dtype == "half" or args.dtype == "fp16":
    dtype = torch.half
elif args.dtype == "bfloat" or args.dtype == "bf16":
    dtype = torch.bfloat16
else:
    raise ValueError("Incorrect value of dtype")

torch.set_default_device(device)
torch.set_default_dtype(dtype)

x = torch.randn(args.n_batch, args.n_seq, args.n_in, requires_grad=False)
y = torch.randn(args.n_batch, args.n_seq, args.n_out, requires_grad=False)

baseline_nflops = (args.n_layer*12-2) * args.n_batch * args.n_seq * args.n_in \
        * args.n_out
mlp = MLPLinear(args.n_layer, args.n_in, args.n_out)
timestats = mytimeit("MLPLinear", baseline_nflops)
print("Flops/linear: 1.0")
print()
rows.append({'note': 'MLPLinear', 'flops/linear': 1.0, **vars(args), \
        **timestats})

mlp = MLPLoRA(args.n_layer, args.n_in, args.n_out, args.n_rank)
timestats = mytimeit("MLPLoRA", 0)
print()
rows.append({'note': 'MLPLoRA', **vars(args), **timestats})

mlp = MLPLightLoRA(args.n_layer, args.n_in, args.n_out, args.n_rank)
timestats = mytimeit("MLPLightLoRA", mlp.flops(x))
print()
rows.append({'note': 'MLPLightLoRA', **vars(args), **timestats})


df = pd.DataFrame.from_records(rows).drop(columns="out")
df.sort_values(['mean_time', 'max_mem_MB'], 
               ascending = [True, True], 
               inplace=True)
df.to_csv(args.out)
print(df)

