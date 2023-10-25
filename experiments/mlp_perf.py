import torch
from lightlora import *
from copy import deepcopy
from time import time
import torch.utils.benchmark as benchmark
from argparse import ArgumentParser

parser = ArgumentParser(prog="Parameters for LightLora")
parser.add_argument("--n_batch", type=int, default=1)
parser.add_argument("--n_seq", type=int, default=4096)
parser.add_argument("--n_layer", type=int, default=1)
parser.add_argument("--n_in", type=int, default=1024)
parser.add_argument("--n_out", type=int, default=1024)
parser.add_argument("--n_rank", type=int, default=32)
parser.add_argument("--dtype", default="float")

args = parser.parse_args()

class MLP(torch.nn.Module):
    def __init__(self, n_layer, n_in, n_out, n_rank, device, dtype):
        super().__init__()
        self.w1 = torch.nn.ParameterList()
        self.u1 = torch.nn.ParameterList()
        self.v1 = torch.nn.ParameterList()
        self.w2 = torch.nn.ParameterList()
        self.u2 = torch.nn.ParameterList()
        self.v2 = torch.nn.ParameterList()
        self.n_layer = n_layer
        for i in range(n_layer):
            self.w1.append(torch.nn.Parameter(torch.randn(n_in, n_out, \
                    device=device, dtype=dtype)))
            self.u1.append(torch.nn.Parameter(torch.randn(n_in, n_rank, \
                    device=device, dtype=dtype)))
            self.v1.append(torch.nn.Parameter(torch.randn(n_rank, n_out, \
                    device=device, dtype=dtype)))
            self.w2.append(torch.nn.Parameter(torch.randn(n_out, n_in, \
                    device=device, dtype=dtype)))
            self.u2.append(torch.nn.Parameter(torch.randn(n_out, n_rank, \
                    device=device, dtype=dtype)))
            self.v2.append(torch.nn.Parameter(torch.randn(n_rank, n_in, \
                    device=device, dtype=dtype)))

    def forward_baseline(self, x):
        for w in self.w1:
            w.requires_grad = True
        for w in self.w2:
            w.requires_grad = True
        y = x
        for i in range(self.n_layer):
            y = (y @ self.w1[i]) @ self.w2[i]
        return y

    def forward_lora_xu(self, x):
        for w in self.w1:
            w.requires_grad = False
        for w in self.w2:
            w.requires_grad = False
        y = x
        for i in range(self.n_layer):
            y = y@self.w1[i] + (y@self.u1[i])@self.v1[i]
            y = y@self.w2[i] + (y@self.u2[i])@self.v2[i]
        return y

    def forward_lora_wpuv(self, x):
        for w in self.w1:
            w.requires_grad = False
        for w in self.w2:
            w.requires_grad = False
        y = x
        for i in range(self.n_layer):
            y = y @ (self.w1[i] + self.u1[i]@self.v1[i])
            y = y @ (self.w2[i] + self.u2[i]@self.v2[i])
        return y

    def forward_light_lora(self, x):
        for w in self.w1:
            w.requires_grad = False
        for w in self.w2:
            w.requires_grad = False
        y = x
        for i in range(self.n_layer):
            y = light_lora.apply(y, self.w1[i], self.u1[i], self.v1[i])
            y = light_lora.apply(y, self.w2[i], self.u2[i], self.v2[i])
        return y

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

device = torch.device("cuda")
if args.dtype == "float" or args.dtype == "fp32":
    dtype = torch.float
elif args.dtype == "half" or args.dtype == "fp16":
    dtype = torch.half
elif args.dtype == "bfloat" or args.dtype == "bf16":
    dtype = torch.bfloat16
else:
    raise ValueError("Incorrect value of dtype")

mlp = MLP(args.n_layer, args.n_in, args.n_out, args.n_rank, device, dtype)
x = torch.randn(args.n_batch, args.n_seq, args.n_in, device=device, \
        dtype=dtype, requires_grad=False)
y = torch.randn(args.n_batch, args.n_seq, args.n_out, device=device, \
        dtype=dtype, requires_grad=False)

light_lora = None
baseline_nflops = args.n_layer * 12 * args.n_batch * args.n_seq * args.n_in \
        * args.n_out

mlp.forward = mlp.forward_baseline
mlp.zero_grad()
mytimeit("baseline", baseline_nflops)
print("Flops/linear: 1.0")
print()

mlp.forward = mlp.forward_lora_xu
mlp.zero_grad()
mytimeit("lora_xu", 0)
print()

mlp.forward = mlp.forward_lora_wpuv
mlp.zero_grad()
mytimeit("lora_wpuv", 0)
print()

for i in range(1, 3):
    for j in range(1, 6):
        print("path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection()(i, j)
        mlp.forward = mlp.forward_light_lora
        mlp.zero_grad()
        light_lora_nflops = args.n_layer * light_lora.flops(x, mlp.w1[0], \
                mlp.u1[0], mlp.v1[0])
        light_lora_nflops += args.n_layer * light_lora.flops(y, mlp.w2[0], \
                mlp.u1[0], mlp.v1[0])
        mytimeit("light_lora", light_lora_nflops)
        print("Flops/linear: {}".format(light_lora_nflops \
                / baseline_nflops))
        print()

for i in range(1, 2):
    for j in range(1, 3):
        print("wpuv path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection_wpuv()(i, j)
        mlp.forward = mlp.forward_light_lora
        mlp.zero_grad()
        light_lora_nflops = args.n_layer * light_lora.flops(x, mlp.w1[0], \
                mlp.u1[0], mlp.v1[0])
        light_lora_nflops += args.n_layer * light_lora.flops(y, mlp.w2[0], \
                mlp.u1[0], mlp.v1[0])
        mytimeit("light_lora", light_lora_nflops)
        print("Flops/linear: {}".format(light_lora_nflops \
                / baseline_nflops))
        print()

for i in range(1, 2):
    for j in range(1, 2):
        print("xu path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection_xu()(i, j)
        mlp.forward = mlp.forward_light_lora
        mlp.zero_grad()
        light_lora_nflops = args.n_layer * light_lora.flops(x, mlp.w1[0], \
                mlp.u1[0], mlp.v1[0])
        light_lora_nflops += args.n_layer * light_lora.flops(y, mlp.w2[0], \
                mlp.u1[0], mlp.v1[0])
        mytimeit("light_lora", light_lora_nflops)
        print("Flops/linear: {}".format(light_lora_nflops \
                / baseline_nflops))
        print()

