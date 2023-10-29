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
parser.add_argument("--short", choices=["True", "False"], default="True")
parser.add_argument('-o', "--out", type=str, default='out')

args = parser.parse_args()
rows = []

def mytimeit(statement, glbls):
    w.grad = None
    x.grad = None
    u.grad = None
    v.grad = None
    bench = benchmark.Timer(stmt=statement, globals=glbls)
    measure_warmup = bench.blocked_autorange(min_run_time=1.0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    measure = bench.blocked_autorange(min_run_time=1.0)
    print("Evaluating \"{}\"".format(statement))
    print("Mean time: {} us".format(measure.mean * 1000000))
    print("Max mem: {} MB".format(torch.cuda.max_memory_allocated()/2**20))
    return {'mean_time': measure.mean * 1000000,
            'max_mem_MB': torch.cuda.max_memory_allocated() / 2**20}

def mytimeit_lightlora(path_f, path_b):
    print("path_f={} path_b={}".format(path_f, path_b))
    global light_lora
    light_lora = light_lora_collection[path_f, path_b]
    glbls={'w': w, 'x': x, 'u': u, 'v': v, 'b': b, 'light_lora': light_lora}
    timestats = mytimeit("light_lora.apply(x, w, u, v, b).sum().backward()", \
            glbls)
    print()
    rows.append({'path_f': path_f, 'path_b': path_b, \
            'note': light_lora.__name__, **timestats})
    return timestats['mean_time']

def mytimeit_lightlora_fwd(path_f, path_b):
    print("path_f={} path_b={}".format(path_f, path_b))
    global light_lora
    light_lora = light_lora_collection[path_f, path_b]
    glbls={'w': w, 'x': x, 'u': u, 'v': v, 'b': b, 'light_lora': light_lora}
    timestats = mytimeit("light_lora.apply(x, w, u, v, b)", glbls)
    print()
    rows.append({'path_f': path_f, 'path_b': path_b, \
            'note': light_lora.__name__, **timestats})
    return timestats['mean_time']

def mytimeit_lightlora_bwd(path_f, path_b):
    print("path_f={} path_b={}".format(path_f, path_b))
    global light_lora
    y = light_lora_collection[path_f, path_b].apply(x, w, u, v, b)
    glbls = {'y': y}
    timestats = mytimeit("y.sum().backward(retain_graph=True)", glbls)
    print()
    rows.append({'path_f': path_f, 'path_b': path_b, \
            'note': light_lora.__name__, **timestats})
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

# Now W shall not accumulate gradient any more
w.requires_grad = False

if b is not None:
    glbls = {'x': x, 'w': w, 'u': u, 'v': v, 'b': b}
    stmt_all = ["(x@w+(x@u)@v+b)", "xx=x.reshape(-1,x.shape[-1]); (b.addmm(" \
            "xx,w).addmm_(xx.mm(u), v).reshape(*x.shape[:-1],w.shape[-1]))", \
            "(x@(w+u@v)+b)", "xx=x.reshape(-1, x.shape[-1]); (b.addmm(xx,w." \
            "addmm(u,v)))"]
else:
    glbls = {'x': x, 'w': w, 'u': u, 'v': v}
    stmt_all = ["(x@w+(x@u)@v)", "xx=x.reshape(-1,x.shape[-1]); (xx.mm(w)" \
            ".addmm_(xx.mm(u),v).reshape(*x.shape[:-1],w.shape[-1]))", \
            "(x@(w+u@v))", "xx=x.reshape(-1,x.shape[-1]); (xx.mm(w.addmm(" \
            "u,v)))"]

for stmt in stmt_all:
    timestats = mytimeit(stmt, glbls)
    print()
    rows.append({'note': stmt, **vars(args), **timestats})
    stmt2 = stmt + ".sum().backward()"
    timestats = mytimeit(stmt2, glbls)
    print()
    rows.append({'note': stmt2, **vars(args), **timestats})

# Find the fastest forward+backward
if args.short == "True":
    fwd_keys = light_lora_collection.forward_keys_short
    bwd_keys = light_lora_collection.backward_keys_short
else:
    fwd_keys = light_lora_collection.forward_keys
    bwd_keys = light_lora_collection.backward_keys
fast_f = fwd_keys[0]
fast_b = bwd_keys[0]
fast_mean = torch.inf
for path_f in fwd_keys:
    mean = mytimeit_lightlora_fwd(path_f, fast_b)
    if fast_mean > mean:
        fast_f = path_f
        fast_mean = mean
fast_mean = torch.inf
for path_b in bwd_keys:
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

