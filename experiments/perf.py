import torch
from lightlora import *
from copy import deepcopy
from time import time
import torch.utils.benchmark as benchmark
from argparse import ArgumentParser

parser = ArgumentParser(prog="Parameters for LightLora")
parser.add_argument("--n_batch", type=int, default=4096)
parser.add_argument("--n_in", type=int, default=1024)
parser.add_argument("--n_out", type=int, default=1024)
parser.add_argument("--n_rank", type=int, default=32)
parser.add_argument("--dtype", default="float")

args = parser.parse_args()

def mytimeit(statement, nflops):
    w.grad = None
    x.grad = None
    u.grad = None
    v.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    b = benchmark.Timer(stmt=statement, globals={'w': w, 'x': x, 'u': u, \
            'v': v, 'light_lora': light_lora, 'torch': torch})
    measure = b.blocked_autorange()
    print("Evaluating \"{}\"".format(statement))
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

w = torch.nn.Parameter(torch.randn(args.n_in, args.n_out, device=device, \
    dtype=dtype), requires_grad=True)
x = torch.randn(args.n_batch, args.n_in, device=device, dtype=dtype, \
        requires_grad=True)
u = torch.randn(args.n_in, args.n_rank, device=device, dtype=dtype, \
        requires_grad=True)
v = torch.randn(args.n_rank, args.n_out, device=device, dtype=dtype, \
        requires_grad=True)

print("x.shape={} w.shape={} u.shape={} v.shape={}".format( \
    x.shape, w.shape, u.shape, v.shape))
print()

light_lora = None
mytimeit("(x@w).sum().backward()", 6*prod(x.shape)*w.shape[1])
print("Flops/linear: 1.0")
print()

# Now W shall not accumulate gradient any more
w.requires_grad = False

mytimeit("(x@w+(x@u)@v).sum().backward()", 0)
print()

mytimeit("(x@(w+u@v)).sum().backward()", 0)
print()

for i in range(1, 3):
    for j in range(1, 6):
        print("path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection()(i, j)
        mytimeit("light_lora.apply(x, w, u, v).sum().backward()", \
                light_lora.flops(x, w, u, v))
        print("Flops/linear: {}".format(light_lora.flops(x, w, u, v) \
                / (6*prod(x.shape)*w.shape[1])))
        print()

for i in range(1, 2):
    for j in range(1, 3):
        print("wpuv path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection_wpuv()(i, j)
        mytimeit("light_lora.apply(x, w, u, v).sum().backward()", \
                light_lora.flops(x, w, u, v))
        print("Flops/linear: {}".format(light_lora.flops(x, w, u, v) \
                / (6*prod(x.shape)*w.shape[1])))
        print()

for i in range(1, 2):
    for j in range(1, 2):
        print("xu path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection_xu()(i, j)
        mytimeit("light_lora.apply(x, w, u, v).sum().backward()", \
                light_lora.flops(x, w, u, v))
        print("Flops/linear: {}".format(light_lora.flops(x, w, u, v) \
                / (6*prod(x.shape)*w.shape[1])))
        print()

