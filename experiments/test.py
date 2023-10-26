import torch
from lightlora import *
from copy import deepcopy

device = torch.device("cuda")
torch.set_default_dtype(torch.double)
w = torch.nn.Parameter(torch.randn(1024, 1024, device=device), requires_grad=False)
x = torch.randn(4, 1024, 1024, device=device, requires_grad=True)
u = torch.randn(1024, 128, device=device, requires_grad=True)
v = torch.randn(128, 1024, device=device, requires_grad=True)
b = torch.randn(1024, device=device, requires_grad=True)
y = x@w + (x@u)@v + b
loss = y.sum()
loss.backward()
x_grad, u_grad, v_grad, b_grad = deepcopy(x.grad), deepcopy(u.grad), \
        deepcopy(v.grad), deepcopy(b.grad)

# Check all lora_collection paths
lora_collection = LightLoRACollection()
path_f = lora_collection.forward_keys[0]
for path_b in lora_collection.backward_keys:
    x.grad, u.grad, v.grad, b.grad = None, None, None, None
    print("x.requires_grad=True path_f={} path_b={}".format(path_f, path_b))
    light_lora = lora_collection[path_f, path_b].apply
    y1 = light_lora(x, w, u, v, b)
    loss = y1.sum()
    print(torch.norm(y1-y) / torch.norm(y))
    loss.backward()
    print(torch.norm(x_grad-x.grad) / torch.norm(x_grad))
    print(torch.norm(u_grad-u.grad) / torch.norm(u_grad))
    print(torch.norm(v_grad-v.grad) / torch.norm(v_grad))
    if b.requires_grad:
        print("b: ", torch.norm(b_grad-b.grad) / torch.norm(b_grad))
    print()
path_b = lora_collection.backward_keys[0]
for path_f in lora_collection.forward_keys[1:]:
    x.grad, u.grad, v.grad = None, None, None
    print("x.requires_grad=True path_f={} path_b={}".format(path_f, path_b))
    light_lora = lora_collection[path_f, path_b].apply
    y1 = light_lora(x, w, u, v, b)
    loss = y1.sum()
    print(torch.norm(y1-y) / torch.norm(y))
    loss.backward()
    print(torch.norm(x_grad-x.grad) / torch.norm(x_grad))
    print(torch.norm(u_grad-u.grad) / torch.norm(u_grad))
    print(torch.norm(v_grad-v.grad) / torch.norm(v_grad))
    if b.requires_grad:
        print("b: ", torch.norm(b_grad-b.grad) / torch.norm(b_grad))
    print()

# Check all lora_nodx_collection paths
lora_nodx_collection = LightLoRANodXCollection()
path_f = lora_nodx_collection.forward_keys[0]
for path_b in lora_nodx_collection.backward_keys:
    x.grad, u.grad, v.grad, b.grad = None, None, None, None
    print("x.requires_grad=False path_f={} path_b={}".format(path_f, path_b))
    light_lora_nodx = lora_nodx_collection[path_f, path_b].apply
    y1 = light_lora_nodx(x, w, u, v, b)
    loss = y1.sum()
    print(torch.norm(y1-y) / torch.norm(y))
    loss.backward()
    print(torch.norm(u_grad-u.grad) / torch.norm(u_grad))
    print(torch.norm(v_grad-v.grad) / torch.norm(v_grad))
    if b.requires_grad:
        print("b: ", torch.norm(b_grad-b.grad) / torch.norm(b_grad))
    print()
path_b = lora_nodx_collection.backward_keys[0]
for path_f in lora_nodx_collection.forward_keys[1:]:
    x.grad, u.grad, v.grad, b.grad = None, None, None, None
    print("x.requires_grad=False path_f={} path_b={}".format(path_f, path_b))
    light_lora = lora_nodx_collection[path_f, path_b].apply
    y1 = light_lora(x, w, u, v, b)
    loss = y1.sum()
    print(torch.norm(y1-y) / torch.norm(y))
    loss.backward()
    print(torch.norm(u_grad-u.grad) / torch.norm(u_grad))
    print(torch.norm(v_grad-v.grad) / torch.norm(v_grad))
    if b.requires_grad:
        print("b: ", torch.norm(b_grad-b.grad) / torch.norm(b_grad))
    print()

