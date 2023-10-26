import torch
from lightlora import *
from copy import deepcopy
from time import time
import torch.utils.benchmark as benchmark
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser(prog="Parameters for LightLora")
parser.add_argument("--n_batch", type=int, default=1)
parser.add_argument("--n_seq", type=int, default=4096)
parser.add_argument("--n_in", type=int, default=1024)
parser.add_argument("--n_out", type=int, default=1024)
parser.add_argument("--n_rank", type=int, default=32)
parser.add_argument("--dtype", default="float")
parser.add_argument('-o', "--out", type=str, default='out')

args = parser.parse_args()
rows = []

def mytimeit(statement, nflops):
    w.grad = None
    x.grad = None
    u.grad = None
    v.grad = None
    b = benchmark.Timer(stmt=statement, globals={'w': w, 'x': x, 'u': u, \
            'v': v, 'light_lora': light_lora, 'torch': torch})
    measure_warmup = b.blocked_autorange(min_run_time=1.0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    measure = b.blocked_autorange(min_run_time=1.0)
    print("Evaluating \"{}\"".format(statement))
    print("Mean time: {} us".format(measure.mean * 1000000))
    print("GFlops: {}".format(nflops*1e-9))
    print("GFlops/s: {}".format(nflops*1e-9 / measure.mean))
    print("Max mem: {} MB".format(torch.cuda.max_memory_allocated()/2**20))
    return {'mean_time': measure.mean * 1000000,
            'Gflops': nflops * 1e-9, 
            'Gflops/s': nflops * 1e-9 / measure.mean,
            'max_mem_MB': torch.cuda.max_memory_allocated() / 2**20}

def mytimeit_lightlora(path_f, path_b):
    print("path_f={} path_b={}".format(path_f, path_b))
    global light_lora
    light_lora = lora_collection[path_f, path_b]
    #light_lora.apply = torch.compile(light_lora.apply)
    flops = light_lora.flops(x, w, u, v)
    flops_linear = flops / (6*prod(x.shape)*w.shape[1])
    timestats = mytimeit("light_lora.apply(x, w, u, v).sum().backward()", 
                         flops)
    print("Flops/linear: {}".format(flops_linear))
    print()
    rows.append({'path_f': path_f, 'path_b': path_b, 'note': light_lora.__name__, \
            'flops': flops, 'flops/linear': flops_linear, **vars(args), \
            **timestats})
    return timestats['mean_time']

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

w = torch.nn.Parameter(torch.randn(args.n_in, args.n_out), requires_grad=True)
x = torch.randn(args.n_batch, args.n_seq, args.n_in, requires_grad=True)
u = torch.randn(args.n_in, args.n_rank, requires_grad=True)
v = torch.randn(args.n_rank, args.n_out, requires_grad=True)

print("x.shape={} w.shape={} u.shape={} v.shape={}".format( \
    x.shape, w.shape, u.shape, v.shape))
print()

light_lora = None
timestats = mytimeit("(x@w).sum().backward()", 6*prod(x.shape)*w.shape[1])
print("Flops/linear: 1.0")
print()
rows.append({'note': 'x@w',
             'flops/linear': 1.0,
             **vars(args), **timestats})

# Now W shall not accumulate gradient any more
w.requires_grad = False

timestats = mytimeit("(x@w+(x@u)@v).sum().backward()", 0)
print()
rows.append({'note': 'x@w+(x@u)@v',
             **vars(args), **timestats})

timestats = mytimeit("(x@(w+u@v)).sum().backward()", 0)
print()
rows.append({'note': 'x@(w+u@v)',
             **vars(args), **timestats})

# Find the fastest forward+backward
lora_collection = LightLoRACollection()
fast_f = lora_collection.forward_keys[0]
fast_b = lora_collection.backward_keys[0]
fast_mean = mytimeit_lightlora(fast_f, fast_b)
for path_f in lora_collection.forward_keys[1:]:
    mean = mytimeit_lightlora(path_f, fast_b)
    if fast_mean > mean:
        fast_f = path_f
        fast_mean = mean
for path_b in lora_collection.backward_keys[1:]:
    mean = mytimeit_lightlora(fast_f, path_b)
    if fast_mean > mean:
        fast_b = path_b
        fast_mean = mean

# Check the fastest forward and backward
mytimeit_lightlora(fast_f, fast_b)

# Disable FW+BW with some saved intermediate results
#for i in range(1, 2):
#    for j in range(1, 3):
#        print("wpuv path_f={} path_b={}".format(i, j))
#        light_lora = LightLoRACollection_wpuv()(i, j)
#        flops = light_lora.flops(x, w, u, v)
#        flops_linear = flops / (6*prod(x.shape)*w.shape[1])
#        timestats = mytimeit("light_lora.apply(x, w, u, v).sum().backward()", 
#                             flops)
#        print("Flops/linear: {}".format(flops_linear))
#        print()
#        rows.append({'path_f': i, 'path_b': j, 'note': light_lora.__name__, \
#                'flops': flops, 'flops/linear': flops_linear, **vars(args), \
#                **timestats})
#
#for i in range(1, 2):
#    for j in range(1, 2):
#        print("xu path_f={} path_b={}".format(i, j))
#        light_lora = LightLoRACollection_xu()(i, j)
#        flops = light_lora.flops(x, w, u, v)
#        flops_linear = flops / (6*prod(x.shape)*w.shape[1])
#        timestats = mytimeit("light_lora.apply(x, w, u, v).sum().backward()", 
#                             flops)
#        print("Flops/linear: {}".format(flops_linear))
#        print()
#        rows.append({'path_f': i, 'path_b': j, 'note': light_lora.__name__, \
#                'flops': flops, 'flops/linear': flops_linear, **vars(args), \
#                **timestats})

df = pd.DataFrame.from_records(rows).drop(columns="out")
df.sort_values(['mean_time', 'max_mem_MB'], 
               ascending = [True, True], 
               inplace=True)
df.to_csv(args.out)
print(df)

