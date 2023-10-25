import torch
from lightlora import *
from copy import deepcopy
from time import time

import torch.utils.benchmark as benchmark
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
w = torch.nn.Parameter(torch.randn(10240, 10240, device=device, \
    dtype=torch.float), requires_grad=True)
x = torch.randn(2560, 10240, device=device, dtype=torch.float, \
        requires_grad=True)
u = torch.randn(10240, 128, device=device, dtype=torch.float, \
        requires_grad=True)
v = torch.randn(128, 10240, device=device, dtype=torch.float, \
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
        light_lora = LightLoRACollection()(i, j).apply
        mytimeit("light_lora(x, w, u, v).sum().backward()", \
                LightLoRACollection()(i, j).flops(x, w, u, v))
        print("Flops/linear: {}".format(LightLoRACollection()(i, j).flops(x, w, u, v) / (6*prod(x.shape)*w.shape[1])))
        print()

for i in range(1, 2):
    for j in range(1, 3):
        print("wpuv path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection_wpuv()(i, j).apply
        #%timeit light_lora(x, w, u, v).sum().backward()
        print()

for i in range(1, 2):
    for j in range(1, 2):
        print("xu path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection_xu()(i, j).apply
        #%timeit light_lora(x, w, u, v).sum().backward()

