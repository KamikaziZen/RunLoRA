import torch
from lightlora import *
from copy import deepcopy

device = torch.device("cuda")
w = torch.nn.Parameter(torch.randn(1024, 1024, device=device), requires_grad=False)
x = torch.randn(4096, 1024, device=device, requires_grad=True)
u = torch.randn(1024, 128, device=device, requires_grad=True)
v = torch.randn(128, 1024, device=device, requires_grad=True)
y = x@w + (x@u)@v
loss = y.sum()
loss.backward()
x_grad, u_grad, v_grad = deepcopy(x.grad), deepcopy(u.grad), deepcopy(v.grad)

for i in range(1, 3):
    for j in range(1, 6):
        x.grad, u.grad, v.grad = None, None, None
        print("path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection()(i, j).apply
        y1 = light_lora(x, w, u, v)
        loss = y1.sum()
        print(torch.norm(y1-y) / torch.norm(y))
        loss.backward()
        print(torch.norm(x_grad-x.grad) / torch.norm(x_grad))
        print(torch.norm(u_grad-u.grad) / torch.norm(u_grad))
        print(torch.norm(v_grad-v.grad) / torch.norm(v_grad))
        print()

for i in range(1, 2):
    for j in range(1, 3):
        x.grad, u.grad, v.grad = None, None, None
        print("wpuv path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection_wpuv()(i, j).apply
        y1 = light_lora(x, w, u, v)
        loss = y1.sum()
        print(torch.norm(y1-y) / torch.norm(y))
        loss.backward()
        print(torch.norm(x_grad-x.grad) / torch.norm(x_grad))
        print(torch.norm(u_grad-u.grad) / torch.norm(u_grad))
        print(torch.norm(v_grad-v.grad) / torch.norm(v_grad))
        print()

for i in range(1, 2):
    for j in range(1, 2):
        x.grad, u.grad, v.grad = None, None, None
        print("xu path_f={} path_b={}".format(i, j))
        light_lora = LightLoRACollection_xu()(i, j).apply
        y1 = light_lora(x, w, u, v)
        loss = y1.sum()
        print(torch.norm(y1-y) / torch.norm(y))
        loss.backward()
        print(torch.norm(x_grad-x.grad) / torch.norm(x_grad))
        print(torch.norm(u_grad-u.grad) / torch.norm(u_grad))
        print(torch.norm(v_grad-v.grad) / torch.norm(v_grad))
        print()
