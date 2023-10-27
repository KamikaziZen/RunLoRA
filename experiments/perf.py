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
parser.add_argument("--b_is_None", choices=["True", "False"], default="True")
parser.add_argument("--x_req_grad", choices=["True", "False"], default="True")
parser.add_argument("--u_req_grad", choices=["True", "False"], default="True")
parser.add_argument("--v_req_grad", choices=["True", "False"], default="True")
parser.add_argument("--b_req_grad", choices=["True", "False"], default="True")
parser.add_argument("--dtype", default="float")
parser.add_argument('-o', "--out", type=str, default='out')

args = parser.parse_args()
rows = []

def mytimeit(statement, nflops):
    w.grad = None
    x.grad = None
    u.grad = None
    v.grad = None
    bench = benchmark.Timer(stmt=statement, globals={'w': w, 'x': x, 'u': u, \
            'v': v, 'b': b, 'light_lora': light_lora, 'torch': torch})
    measure_warmup = bench.blocked_autorange(min_run_time=1.0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    measure = bench.blocked_autorange(min_run_time=1.0)
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
    light_lora = light_lora_collection[path_f, path_b]
    #light_lora.apply = torch.compile(light_lora.apply)
    flops = light_lora.flops(x, w, u, v, b)
    flops_linear = flops / baseline_nflops
    timestats = mytimeit("light_lora.apply(x, w, u, v, b).sum().backward()", 
                         flops)
    print("Flops/linear: {}".format(flops_linear))
    print()
    rows.append({'path_f': path_f, 'path_b': path_b, \
            'note': light_lora.__name__, 'flops': flops, \
            'flops/linear': flops_linear, **vars(args), \
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
x_req_grad = (args.x_req_grad == "True")
u_req_grad = (args.u_req_grad == "True")
v_req_grad = (args.v_req_grad == "True")
b_req_grad = (args.b_req_grad == "True")
x = torch.randn(args.n_batch, args.n_seq, args.n_in, requires_grad=x_req_grad)
u = torch.randn(args.n_in, args.n_rank, requires_grad=u_req_grad)
v = torch.randn(args.n_rank, args.n_out, requires_grad=v_req_grad)
if args.b_is_None == "True":
    b = None
else:
    b = torch.randn(args.n_out, requires_grad=b_req_grad)

print("x.shape={} w.shape={} u.shape={} v.shape={} b.shape={}".format( \
    x.shape, w.shape, u.shape, v.shape, b.shape if b is not None else None))
print()

light_lora = None
if x.requires_grad:
    baseline_nflops = 6 * prod(x.shape) * w.shape[1]
else:
    baseline_nflops = 4 * prod(x.shape) * w.shape[1]

if b is not None:
    timestats = mytimeit("(x@w+b).sum().backward()", baseline_nflops)
    print("Flops/linear: 1.0")
    print()
    rows.append({'note': 'x@w+b',
                 'flops/linear': 1.0,
                 **vars(args), **timestats})
else:
    timestats = mytimeit("(x@w).sum().backward()", baseline_nflops)
    print("Flops/linear: 1.0")
    print()
    rows.append({'note': 'x@w',
                 'flops/linear': 1.0,
                 **vars(args), **timestats})

# Now W shall not accumulate gradient any more
w.requires_grad = False

if b is not None:
    timestats = mytimeit("(x@w+(x@u)@v+b).sum().backward()", 0)
    print()
    rows.append({'note': 'x@w+(x@u)@v+b',
                 **vars(args), **timestats})
else:
    timestats = mytimeit("(x@w+(x@u)@v).sum().backward()", 0)
    print()
    rows.append({'note': 'x@w+(x@u)@v',
                 **vars(args), **timestats})

if b is not None:
    timestats = mytimeit("(x@(w+u@v)+b).sum().backward()", 0)
    print()
    rows.append({'note': 'x@(w+u@v)+b',
                 **vars(args), **timestats})
else:
    timestats = mytimeit("(x@(w+u@v)).sum().backward()", 0)
    print()
    rows.append({'note': 'x@(w+u@v)',
                 **vars(args), **timestats})

# Find the fastest forward+backward
fast_f = light_lora_collection.forward_keys[0]
fast_b = light_lora_collection.backward_keys[0]
fast_mean = mytimeit_lightlora(fast_f, fast_b)
for path_f in light_lora_collection.forward_keys[1:]:
    mean = mytimeit_lightlora(path_f, fast_b)
    if fast_mean > mean:
        fast_f = path_f
        fast_mean = mean
for path_b in light_lora_collection.backward_keys[1:]:
    mean = mytimeit_lightlora(fast_f, path_b)
    if fast_mean > mean:
        fast_b = path_b
        fast_mean = mean

df = pd.DataFrame.from_records(rows).drop(columns="out")
df.sort_values(['mean_time', 'max_mem_MB'], 
               ascending = [True, True], 
               inplace=True)
df.to_csv(args.out)
print(args)
print(df.drop(columns=["n_batch", "n_seq", "n_in", "n_out", "n_rank", \
        "x_req_grad", "u_req_grad", "v_req_grad", "b_req_grad", "dtype"]))

