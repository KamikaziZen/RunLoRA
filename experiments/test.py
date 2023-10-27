import torch
from lightlora import *
from copy import deepcopy

device = torch.device("cuda")
torch.set_default_device(device)
torch.set_default_dtype(torch.double)
x = torch.randn(4, 1280, 1024, requires_grad=True)
w = torch.randn(1024, 768, requires_grad=False)
u = torch.randn(1024, 128, requires_grad=True)
v = torch.randn(128, 768, requires_grad=True)
b = torch.randn(768, requires_grad=True)
y = x@w + (x@u)@v + b
loss = y.sum()
loss.backward()
x_grad, u_grad, v_grad, b_grad = deepcopy(x.grad), deepcopy(u.grad), \
        deepcopy(v.grad), deepcopy(b.grad)

# Check all forward light_lora_collection paths
path_b = light_lora_collection.backward_keys[0]
for path_f in light_lora_collection.forward_keys:
    print("Check forward path_f={}".format(path_f))
    light_lora = light_lora_collection[path_f, path_b].apply
    y1 = light_lora(x, w, u, v, b)
    print("y: ", torch.norm(y1-y) / torch.norm(y))
    print()

# Check all backward light_lora_collection paths for all possible requires_grad
# flags for x, w, u, v and b
for req_grad_flags in range(1, 2**4):
    x.requires_grad = bool(req_grad_flags & 1)
    u.requires_grad = bool(req_grad_flags & 2)
    v.requires_grad = bool(req_grad_flags & 4)
    b.requires_grad = bool(req_grad_flags & 8)
    path_f = light_lora_collection.forward_keys[0]
    for path_b in light_lora_collection.backward_keys:
        x.grad, u.grad, v.grad, b.grad = None, None, None, None
        print("Check backward path_f={} path_b={}".format(path_f, path_b))
        print("x.requires_grad={} u.requires_grad={} v.requires_grad={} "
                "b_requires_grad={}".format(x.requires_grad, u.requires_grad, \
                        v.requires_grad, b.requires_grad))
        light_lora = light_lora_collection[path_f, path_b].apply
        y1 = light_lora(x, w, u, v, b)
        loss = y1.sum()
        loss.backward()
        if x.requires_grad:
            print("dx: ", torch.norm(x_grad-x.grad) / torch.norm(x_grad))
        if u.requires_grad:
            print("du: ", torch.norm(u_grad-u.grad) / torch.norm(u_grad))
        if v.requires_grad:
            print("dv: ", torch.norm(v_grad-v.grad) / torch.norm(v_grad))
        if b.requires_grad:
            print("db: ", torch.norm(b_grad-b.grad) / torch.norm(b_grad))
        print()

# Now b is None
b = None
x.grad, u.grad, v.grad = [None] * 3
y = x@w + (x@u)@v
loss = y.sum()
loss.backward()
x_grad, u_grad, v_grad = deepcopy(x.grad), deepcopy(u.grad), deepcopy(v.grad)

# Check all forward light_lora_collection paths
path_b = light_lora_collection.backward_keys[0]
for path_f in light_lora_collection.forward_keys:
    print("Check forward path_f={}".format(path_f))
    light_lora = light_lora_collection[path_f, path_b].apply
    y1 = light_lora(x, w, u, v, b)
    print("y: ", torch.norm(y1-y) / torch.norm(y))
    print()

# Check all backward light_lora_collection paths for all possible requires_grad
# flags for x, w, u, v and b
for req_grad_flags in range(1, 2**3):
    x.requires_grad = bool(req_grad_flags & 1)
    u.requires_grad = bool(req_grad_flags & 2)
    v.requires_grad = bool(req_grad_flags & 4)
    path_f = light_lora_collection.forward_keys[0]
    for path_b in light_lora_collection.backward_keys:
        x.grad, u.grad, v.grad = None, None, None
        print("Check backward path_f={} path_b={}".format(path_f, path_b))
        print("x.requires_grad={} u.requires_grad={} v.requires_grad={}" \
                .format(x.requires_grad, u.requires_grad, \
                        v.requires_grad))
        light_lora = light_lora_collection[path_f, path_b].apply
        y1 = light_lora(x, w, u, v, b)
        loss = y1.sum()
        loss.backward()
        if x.requires_grad:
            print("dx: ", torch.norm(x_grad-x.grad) / torch.norm(x_grad))
        if u.requires_grad:
            print("du: ", torch.norm(u_grad-u.grad) / torch.norm(u_grad))
        if v.requires_grad:
            print("dv: ", torch.norm(v_grad-v.grad) / torch.norm(v_grad))
        print()

