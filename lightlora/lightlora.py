import torch
from torch.utils import benchmark
import torch.nn as nn
from math import prod
from collections import defaultdict


def timeit_lightlora(paths_f, paths_b, X, W, U, V, B):
    x = torch.zeros_like(X, requires_grad=X.requires_grad)
    w = torch.zeros_like(W, requires_grad=W.requires_grad)
    u = torch.zeros_like(U, requires_grad=U.requires_grad)
    v = torch.zeros_like(V, requires_grad=V.requires_grad)
    if B is not None:
        b = torch.zeros_like(B, requires_grad=B.requires_grad)
    else:
        b = None

    # Benchmark forward
    statement = "light_lora.apply(x, w, u, v, b)"
    path_b = paths_b[0]
    best_path_f = -1
    best_path_time = torch.inf
    for path_f in paths_f:
        x.grad, w.grad, u.grad, v.grad = [None] * 4
        if b is not None:
            b.grad = None
        light_lora = light_lora_collection[path_f, path_b]
        globals_ = {'light_lora': light_lora, 'x': x,
                    'w': w, 'u': u, 'v': v, 'b': b}
        bench = benchmark.Timer(stmt=statement, globals=globals_)
        _ = bench.blocked_autorange(min_run_time=1)
        measure = bench.blocked_autorange(min_run_time=5)
        print(f"Mean time in us for {path_f}: {measure.mean}")
        if best_path_time > measure.mean:
            best_path_time = measure.mean
            best_path_f = path_f
    print()

    # Benchmark backward
    statement = "loss.backward(retain_graph=True)"
    path_f = best_path_f
    best_path_time = torch.inf
    for path_b in paths_b:
        x.grad, w.grad, u.grad, v.grad = [None] * 4
        if b is not None:
            b.grad = None
        light_lora = light_lora_collection[path_f, path_b]
        loss = light_lora.apply(x, w, u, v, b).sum().requires_grad_(True)
        globals_ = {'loss': loss}
        bench = benchmark.Timer(stmt=statement, globals=globals_)
        _ = bench.blocked_autorange(min_run_time=1)
        measure = bench.blocked_autorange(min_run_time=5)
        print(f"Mean time in us for {path_b}: {measure.mean}")
        if best_path_time > measure.mean:
            best_path_time = measure.mean
            best_path_b = path_b
    print()

    return best_path_f, best_path_b


class LightLoRACollection(object):
    def __init__(self):
        self.forward_keys = [i for i in dir(self) \
                if i.startswith("forward") and i[-5:] != "flops"]
        self.backward_keys = [i for i in dir(self) \
                if i.startswith("backward") and i[-5:] != "flops"]
        self.forward_keys_short = ["forward{}".format(i) \
                for i in range(1, 5)]
        self.backward_keys_short = ["backward{}".format(i) \
                for i in range(1, 6)]

        self.flops_benchmarks = {}
        self.time_benchmarks = {}
        self.time_benchmarks_short = {}

    def __getitem__(self, index):
        path_f, path_b = index
        if path_f not in self.forward_keys:
            raise ValueError("Incorrect path_f")
        if path_b not in self.backward_keys:
            raise ValueError("Incorrect path_b")
        path_f_flops = path_f[:8] + "_flops"
        path_b_flops = path_b[:9] + "_flops"
        method_forward = getattr(self, path_f)
        method_backward = getattr(self, path_b)
        method_forward_flops = getattr(self, path_f_flops)
        method_backward_flops = getattr(self, path_b_flops)

        class LightLoRA(torch.autograd.Function):
            forward = method_forward
            backward = method_backward

            def flops(input, W, U, V, b):
                return method_forward_flops(input, W, U, V, b) \
                    + method_backward_flops(input, W, U, V, b)
        return LightLoRA

    def optimize_for_model(self, model, n_batch, lora_r, target_modules, criterions):

        light_lora_mapping = defaultdict(dict)

        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(trgt in module_name for trgt in target_modules):
                continue

            key = (
                torch.Size([n_batch, model.config.max_sequence_length, module.in_features]), False,
                torch.Size([module.in_features, module.out_features]), False,
                torch.Size([module.in_features, lora_r]), True,
                torch.Size([lora_r, module.out_features]), True,
            )

            for criterion in criterions:
                if not self.lookup_best(criterion, key):
                    print(f'Did not find criterion {criterion} for {module_name}. Calculating')
                    w = nn.Parameter(torch.randn(module.in_features, module.out_features), requires_grad=False)
                    x = torch.randn(n_batch, model.config.max_sequence_length, module.in_features, requires_grad=False)
                    u = torch.randn(module.in_features, lora_r, requires_grad=True)
                    v = torch.randn(lora_r, module.out_features, requires_grad=True)
                    _ = self.get_best(criterion, x, w, u, v)

                light_lora_mapping[criterion][module_name] = \
                    self.lookup_best(criterion, key)

        return light_lora_mapping

    def get_best_by_flops(self, X, W, U, V, b):
        if b is not None:
            key = (X.shape, X.requires_grad, W.shape, W.requires_grad,
                   U.shape, U.requires_grad, V.shape, V.requires_grad,
                   b.shape, b.requires_grad)
        else:
            key = (X.shape, X.requires_grad, W.shape, W.requires_grad,
                   U.shape, U.requires_grad, V.shape, V.requires_grad)

        if key not in self.flops_benchmarks:
            path_f_flops = [getattr(self, key+"_flops")(X, W, U, V, b) \
                    for key in self.forward_keys_short]
            path_f_index = 0
            for i in range(len(path_f_flops)):
                print(f'Flops for {self.forward_keys_short[i]}: {path_f_flops[i]}')
                if path_f_flops[path_f_index] > path_f_flops[i]:
                    path_f_index = i
            print()

            path_b_flops = [getattr(self, key+"_flops")(X, W, U, V, b) \
                    for key in self.backward_keys_short]
            path_b_index = 0
            for i in range(len(path_b_flops)):
                print(f'Flops for {self.backward_keys_short[i]}: {path_b_flops[i]}')
                if path_b_flops[path_b_index] > path_b_flops[i]:
                    path_b_index = i
            print()

            self. flops_benchmarks[key] = (
                self.forward_keys_short[path_f_index],
                self.backward_keys_short[path_b_index]
            )

        return self.flops_benchmarks[key]

    def get_best_by_time(self, X, W, U, V, b):
        if b is not None:
            key = (X.shape, X.requires_grad, W.shape, W.requires_grad,
                   U.shape, U.requires_grad, V.shape, V.requires_grad,
                   b.shape, b.requires_grad)
        else:
            key = (X.shape, X.requires_grad, W.shape, W.requires_grad,
                   U.shape, U.requires_grad, V.shape, V.requires_grad)
        if key not in self.time_benchmarks:
            path_f, path_b = timeit_lightlora(self.forward_keys,
                                              self.backward_keys,
                                              X, W, U, V, b)
            self.time_benchmarks[key] = (path_f, path_b)
        return self.time_benchmarks[key]

    def get_best_by_time_short(self, X, W, U, V, b):
        if b is not None:
            key = (X.shape, X.requires_grad, W.shape, W.requires_grad,
                   U.shape, U.requires_grad, V.shape, V.requires_grad,
                   b.shape, b.requires_grad)
        else:
            key = (X.shape, X.requires_grad, W.shape, W.requires_grad,
                   U.shape, U.requires_grad, V.shape, V.requires_grad)
        if key not in self.time_benchmarks_short:
            path_f, path_b = timeit_lightlora(
                self.forward_keys_short, self.backward_keys_short,
                X, W, U, V, b)
            self.time_benchmarks_short[key] = (path_f, path_b)
        return self.time_benchmarks_short[key]

    def get_best(self, criterion, x, W, U, V, b=None):
        if criterion == "flops":
            path_f, path_b = self.get_best_by_flops(x, W, U, V, b)
        elif criterion == "time":
            path_f, path_b = self.get_best_by_time(x, W, U, V, b)
        elif criterion == "time_short":
            path_f, path_b = self.get_best_by_time_short(x, W, U, V, b)

        return self.__getitem__((path_f, path_b))

    def lookup_best(self, criterion, key):
        try:
            if criterion == 'flops':
                path_f, path_b = self.flops_benchmarks[key]
            elif criterion == 'time':
                path_f, path_b = self.time_benchmarks[key]
            elif criterion == 'time_short':
                path_f, path_b = self.time_benchmarks_short[key]
            else:
                raise ValueError(f'Invalid criterion: {criterion}')
        except KeyError:
            return None

        return self.__getitem__((path_f, path_b))

    @staticmethod
    def save_context(ctx, input, W, U, V):
        if U.requires_grad or V.requires_grad:
            save_X = input
        else:
            save_X = torch.empty_like(input, device="meta")
        if input.requires_grad:
            save_W = W
        else:
            save_W = None
        if input.requires_grad or V.requires_grad:
            save_U = U
        else:
            save_U = None
        if input.requires_grad or U.requires_grad:
            save_V = V
        else:
            save_V = None
        ctx.save_for_backward(save_X, save_W, save_U, save_V)

    @staticmethod
    def forward1(ctx, input, W, U, V, b):
        """Y=b+XW+(XU)V save(X,W,U,V)"""
        X = input.contiguous().view(-1, input.shape[-1])
        Y_shape = torch.Size(list(input.shape[:-1]) + [W.shape[1]])
        __class__.save_context(ctx, input, W, U, V)
        #ctx.save_for_backward(input, W, U, V, b)
        if b is not None:
            return (b.addmm(X, W).addmm_(X.mm(U), V)).view(Y_shape)
        else:
            return (X.mm(W).addmm_(X.mm(U), V)).view(Y_shape)

    @staticmethod
    def forward1_flops(input, W, U, V, b):
        nflops = 0
        # input .mm (W)
        nflops += 2 * prod(input.shape) * W.shape[1]
        # input .mm (U)
        nflops += 2 * prod(input.shape) * U.shape[1]
        # (input.mm(W)) .addmm_ (input.mm(U), V)
        nflops += 2 * prod(input.shape[:-1]) * U.shape[1] * V.shape[1]
        return nflops

    @staticmethod
    def forward2(ctx, input, W, U, V, b):
        """Y=b+X(W+UV) save(X,W,U,V)"""
        X = input.contiguous().view(-1, input.shape[-1])
        Y_shape = torch.Size(list(input.shape[:-1]) + [W.shape[1]])
        __class__.save_context(ctx, input, W, U, V)
        if b is not None:
            return b.addmm(X, W.addmm(U, V)).view(Y_shape)
        else:
            return X.mm(W.addmm(U, V)).view(Y_shape)

    @staticmethod
    def forward2_flops(input, W, U, V, b):
        nflops = 0
        # W .addmm (U, V)
        nflops += 2 * U.shape[0] * U.shape[1] * V.shape[1]
        # input .mm (W.addmm(U, V))
        nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops

    @staticmethod
    def forward3(ctx, input, W, U, V, b):
        """Y=b+(XU)V+XW save(X,W,U,V)"""
        X = input.contiguous().view(-1, input.shape[-1])
        Y_shape = torch.Size(list(input.shape[:-1]) + [W.shape[1]])
        __class__.save_context(ctx, input, W, U, V)
        if b is not None:
            return b.addmm(X.mm(U), V).addmm_(X, W).view(Y_shape)
        else:
            return X.mm(U).mm(V).addmm_(X, W).view(Y_shape)

    @staticmethod
    def forward3_flops(input, W, U, V, b):
        nflops = 0
        # input .mm (W)
        nflops += 2 * prod(input.shape) * W.shape[1]
        # input .mm (U)
        nflops += 2 * prod(input.shape) * U.shape[1]
        # (input.mm(W)) .addmm_ (input.mm(U), V)
        nflops += 2 * prod(input.shape[:-1]) * U.shape[1] * V.shape[1]
        return nflops

    @staticmethod
    def forward4(ctx, input, W, U, V, b):
        """Y=b+X(W+UV) save(X,W,U,V)"""
        __class__.save_context(ctx, input, W, U, V)
        if b is not None:
            return b + input @ W.addmm(U, V)
        else:
            return input @ (W.addmm(U, V))

    @staticmethod
    def forward4_flops(input, W, U, V, b):
        nflops = 0
        # W .addmm (U, V)
        nflops += 2 * U.shape[0] * U.shape[1] * V.shape[1]
        # input .mm (W.addmm(U, V))
        nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops
    
    @staticmethod
    def backward1(ctx, grad_output):
        """load(X,W,U,V) Z=dYV'
        dU=X'Z dV=(XU)'dY dX=dYW'+ZU' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X = input.contiguous().view(-1, input.shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        if V_req_grad:
            grad_V = (X.mm(U)).t().mm(dY)
        if X_req_grad:
            grad_input = dY.mm(W.t()).addmm_(Z1, U.t()).view(input.shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward1_X_Z1_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU
        dU=X'Z1 dV=Z2'dY dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.mm(U)
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input = Z1.mm(U.t())
        del Z1
        if V_req_grad:
            grad_V = Z2.t().mm(dY)
        del Z2
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward1_X_Z2_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU
        dU=X'Z1 dV=Z2'dY dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.mm(U)
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if V_req_grad:
            grad_V = Z2.t().mm(dY)
        del Z2
        if X_req_grad:
            grad_input = Z1.mm(U.t())
        del Z1
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward1_X_Z2_dY_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU
        dU=X'Z1 dV=Z2'dY dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.mm(U)
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if V_req_grad:
            grad_V = Z2.t().mm(dY)
        del Z2
        if X_req_grad:
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward1_Z1_X_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU
        dU=X'Z1 dV=Z2'dY dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        if X_req_grad:
            grad_input = Z1.mm(U.t())
        del Z1
        if V_req_grad:
            Z2 = X.mm(U)
        del X, input
        if V_req_grad:
            grad_V = Z2.t().mm(dY)
        del Z2
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward1_Z2_X_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU
        dU=X'Z1 dV=Z2'dY dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if V_req_grad:
            Z2 = X.mm(U)
            grad_V = Z2.t().mm(dY)
        del Z2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input = Z1.mm(U.t())
        del Z1
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward1_Z2_X_dY_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU
        dU=X'Z1 dV=Z2'dY dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if V_req_grad:
            Z2 = X.mm(U)
            grad_V = Z2.t().mm(dY)
        del Z2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward1_flops(input, W, U, V, b):
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                input.requires_grad, W.requires_grad, U.requires_grad, \
                V.requires_grad, b.requires_grad if b is not None else None
        nflops = 0
        if X_req_grad or U_req_grad:
            # Z = dY .mm (V.t())
            nflops += 2 * prod(input.shape[:-1]) * V.shape[1] * V.shape[0]
        if U_req_grad:
            # grad_U = X.t().mm(Z)
            nflops += 2 * prod(input.shape) * V.shape[0]
        if V_req_grad:
            # grad_V = (XU).t().mm(dY)
            nflops += 2 * prod(input.shape) * U.shape[1]
            nflops += 2 * prod(input.shape[:-1]) * prod(V.shape)
        if X_req_grad:
            # grad_input = Z.mm(U.t()).addmm(dY, W.t())
            nflops += 2 * prod(input.shape) * V.shape[0]
            nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops

    @staticmethod
    def backward2(ctx, grad_output):
        """load(X,W,U,V) Z=dYV'
        dU=X'Z dV=U'X'dY dX=dYW'.addmm(Z,U') db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X = input.contiguous().view(-1, input.shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        if V_req_grad:
            grad_V = U.t().mm(X.t().mm(dY))
        if X_req_grad:
            grad_input = dY.mm(W.t()).addmm_(Z1, U.t()).view(input.shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_X_dY_Z1_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.t().mm(dY)
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_X_dY_Z2_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.t().mm(dY)
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_X_Z1_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.t().mm(dY)
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input = Z1.mm(U.t())
        del Z1
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_X_Z1_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.t().mm(dY)
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input = Z1.mm(U.t())
        del Z1
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_X_Z2_dY_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.t().mm(dY)
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_X_Z2_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.t().mm(dY)
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            grad_input = Z1.mm(U.t())
        del Z1
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_dY_X_Z1_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.t().mm(dY)
        if X_req_grad:
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_dY_X_Z2_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.t().mm(dY)
        if X_req_grad:
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_dY_Z2_X_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if V_req_grad:
            Z2 = X.t().mm(dY)
        if X_req_grad:
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_Z1_X_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        if X_req_grad:
            grad_input = Z1.mm(U.t())
        del Z1
        if V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_Z1_X_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        if X_req_grad:
            grad_input = Z1.mm(U.t())
        del Z1
        if V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_Z2_X_dY_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if V_req_grad:
            Z2 = X.t().mm(dY)
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_Z2_X_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if V_req_grad:
            Z2 = X.t().mm(dY)
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input = Z1.mm(U.t())
        del Z1
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_Z2_dY_X_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if V_req_grad:
            Z2 = X.t().mm(dY)
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad or U_req_grad:
            Z1 = dY.mm(V.t())
        if X_req_grad:
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if U_req_grad:
            grad_U = X.t().mm(Z1)
        del X, input
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward2_flops(input, W, U, V, b):
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                input.requires_grad, W.requires_grad, U.requires_grad, \
                V.requires_grad, b.requires_grad if b is not None else None
        nflops = 0
        if X_req_grad or U_req_grad:
            # Z = dY .mm (V.t())
            nflops += 2 * prod(input.shape[:-1]) * V.shape[1] * V.shape[0]
        if U_req_grad:
            # grad_U = X.t().mm (Z)
            nflops += 2 * prod(input.shape) * V.shape[0]
        if V_req_grad:
            # grad_V = U.t() .mm (X.t()).mm(dY)
            nflops += 2 * prod(input.shape) * W.shape[1]
            nflops += 2 * U.shape[0] * U.shape[1] * W.shape[1]
        if X_req_grad:
            # grad_input += dY .mm (W.t()).addmm(Z, U.t())
            nflops += 2 * prod(input.shape) * W.shape[1]
            nflops += 2 * prod(input.shape) * V.shape[0]
        return nflops

    @staticmethod
    def backward3(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X = input.contiguous().view(-1, input.shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        if U_req_grad or V_req_grad:
            Z = X.t().mm(dY)
        if U_req_grad:
            grad_U = Z.mm(V.t())
        if V_req_grad:
            grad_V = (U.t()).mm(Z)
        if X_req_grad:
            grad_input = dY.mm(W.t()).addmm_(dY.mm(V.t()), U.t()).view(input.shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward3_X_dY_Z1_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if X_req_grad:
            Z1 = dY.mm(V.t())
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward3_X_dY_Z2_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if X_req_grad:
            Z1 = dY.mm(V.t())
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward3_X_Z1_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if X_req_grad:
            Z1 = dY.mm(V.t())
            grad_input = Z1.mm(U.t())
        del Z1
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward3_X_Z1_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if X_req_grad:
            Z1 = dY.mm(V.t())
            grad_input = Z1.mm(U.t())
        del Z1
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward3_X_Z2_dY_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            Z1 = dY.mm(V.t())
            grad_input = dY.mm(W.t())
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if X_req_grad:
            grad_input.addmm_(Z1, U.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward3_X_Z2_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            Z1 = dY.mm(V.t())
            grad_input = Z1.mm(U.t())
        del Z1
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward3_Z1_X_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad:
            Z1 = dY.mm(V.t())
            grad_input = Z1.mm(U.t())
        del Z1
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward3_Z1_X_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYW'+Z1U' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad:
            Z1 = dY.mm(V.t())
            grad_input = Z1.mm(U.t())
        del Z1
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = U.t().mm(Z2)
        del Z2
        if X_req_grad:
            grad_input.addmm_(dY, W.t())
            grad_input = grad_input.view(X_shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward3_flops(input, W, U, V, b):
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                input.requires_grad, W.requires_grad, U.requires_grad, \
                V.requires_grad, b.requires_grad if b is not None else None
        nflops = 0
        # Z2 = X.t() .mm (dY)
        if U_req_grad or V_req_grad:
            nflops += 2 * prod(input.shape) * W.shape[1]
        # Z1 = dY .mm (V.t())
        if X_req_grad:
            nflops += 2 * prod(input.shape[:-1]) * V.shape[1] * V.shape[0]
        # grad_input = dY .mm (W.t())
        if X_req_grad:
            nflops += 2 * prod(input.shape) * W.shape[1]
        # grad_input += Z1 .mm (U.t())
        if X_req_grad:
            nflops += 2 * prod(input.shape[:-1]) * U.shape[1] * U.shape[0]
        # grad_U = Z2 .mm (V.t())
        if U_req_grad:
            nflops += 2 * input.shape[-1] * V.shape[1] * V.shape[0]
        # grad_V = (U.t()) .mm (Z2)
        if V_req_grad:
            nflops += 2 * U.shape[1] * U.shape[0] * W.shape[1]
        return nflops

    @staticmethod
    def backward4(ctx, grad_output):
        """load(X,W,U,V) Z1=W+UV Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYZ1' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X = input.contiguous().view(-1, input.shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        if U_req_grad or V_req_grad:
            Z = X.t().mm(dY)
        if U_req_grad:
            grad_U = Z.mm(V.t())
        if V_req_grad:
            grad_V = (U.t()).mm(Z)
        if X_req_grad:
            grad_input = dY.mm((W.addmm(U, V)).t()).view(input.shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward4_X_Z1_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=W+UV Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYZ1' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if X_req_grad:
            Z1 = W.addmm(U, V)
            grad_input = dY.mm(Z1.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = (U.t()).mm(Z2)
        del Z2
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward4_X_Z2_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=W+UV Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYZ1' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = (U.t()).mm(Z2)
        del Z2
        if X_req_grad:
            Z1 = W.addmm(U, V)
            grad_input = dY.mm(Z1.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward4_Z1_X_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=W+UV Z2=X'dY
        dU=Z2V' dV=U'Z2 dX=dYZ1' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2 = [None] * 2
        if X_req_grad:
            Z1 = W.addmm(U, V)
            grad_input = dY.mm(Z1.t())
            grad_input = grad_input.view(X_shape)
        del Z1
        if U_req_grad or V_req_grad:
            Z2 = X.t().mm(dY)
        del X, input
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        if U_req_grad:
            grad_U = Z2.mm(V.t())
        if V_req_grad:
            grad_V = (U.t()).mm(Z2)
        del Z2
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward4_flops(input, W, U, V, b):
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                input.requires_grad, W.requires_grad, U.requires_grad, \
                V.requires_grad, b.requires_grad if b is not None else None
        nflops = 0
        # Z2 = X.t() .mm (dY)
        if U_req_grad or V_req_grad:
            nflops += 2 * prod(input.shape) * W.shape[1]
        # Z1 = W .addmm (U, V)
        if X_req_grad:
            nflops += 2 * U.shape[0] * U.shape[1] * V.shape[1]
        # grad_input = dY .mm (Z1.t())
        if X_req_grad:
            nflops += 2 * prod(input.shape) * W.shape[1]
        # grad_U = Z2 .mm (V.t())
        if U_req_grad:
            nflops += 2 * W.shape[0] * V.shape[1] * V.shape[0]
        # grad_V = (U.t()) .mm (Z2)
        if V_req_grad:
            nflops += 2 * U.shape[1] * U.shape[0] * W.shape[1]
        return nflops

    @staticmethod
    def backward5(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU Z3=W+UV
        dU=X'Z1 dV=Z2'dY dX=dYZ3' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X = input.contiguous().view(-1, input.shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        if U_req_grad:
            grad_U = X.t().mm(dY.mm(V.t()))
        if V_req_grad:
            grad_V = (X.mm(U)).t().mm(dY)
        if X_req_grad:
            grad_input = dY.mm((W.addmm(U, V)).t()).view(input.shape)
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward5_Z1_X_Z2_Z3_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU Z3=W+UV dU=X'Z1 dV=Z2'dY dX=dYZ3' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2, Z3 = [None] * 3
        if U_req_grad:
            Z1 = dY.mm(V.t())
            grad_U = X.t().mm(Z1)
        del Z1
        if V_req_grad:
            Z2 = X.mm(U)
        del X, input
        if V_req_grad:
            grad_V = Z2.t().mm(dY)
        del Z2
        if X_req_grad:
            Z3 = W.addmm(U, V)
            grad_input = dY.mm(Z3.t())
            grad_input = grad_input.view(X_shape)
        del Z3
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward5_Z1_X_Z3_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU Z3=W+UV
        dU=X'Z1 dV=Z2'dY dX=dYZ3' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2, Z3 = [None] * 3
        if U_req_grad:
            Z1 = dY.mm(V.t())
            grad_U = X.t().mm(Z1)
        del Z1
        if V_req_grad:
            Z2 = X.mm(U)
        del X, input
        if X_req_grad:
            Z3 = W.addmm(U, V)
            grad_input = dY.mm(Z3.t())
            grad_input = grad_input.view(X_shape)
        del Z3
        if V_req_grad:
            grad_V = Z2.t().mm(dY)
        del Z2
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward5_Z1_Z3_X_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU Z3=W+UV
        dU=X'Z1 dV=Z2'dY dX=dYZ3' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2, Z3 = [None] * 3
        if U_req_grad:
            Z1 = dY.mm(V.t())
            grad_U = X.t().mm(Z1)
        del Z1
        if X_req_grad:
            Z3 = W.addmm(U, V)
            grad_input = dY.mm(Z3.t())
            grad_input = grad_input.view(X_shape)
        del Z3
        if V_req_grad:
            Z2 = X.mm(U)
        del X, input
        if V_req_grad:
            grad_V = Z2.t().mm(dY)
        del Z2
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward5_Z3_Z1_X_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU Z3=W+UV
        dU=X'Z1 dV=Z2'dY dX=dYZ3' db=dY.sum(axis=0)"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.contiguous().view(-1, X_shape[-1])
        dY = grad_output.contiguous().view(-1, grad_output.shape[-1])
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
            ctx.needs_input_grad
        grad_input, grad_W, grad_U, grad_V, grad_b = [None] * 5
        Z1, Z2, Z3 = [None] * 3
        if X_req_grad:
            Z3 = W.addmm(U, V)
            grad_input = dY.mm(Z3.t())
            grad_input = grad_input.view(X_shape)
        del Z3
        if U_req_grad:
            Z1 = dY.mm(V.t())
            grad_U = X.t().mm(Z1)
        del Z1
        if V_req_grad:
            Z2 = X.mm(U)
        del X, input
        if V_req_grad:
            grad_V = Z2.t().mm(dY)
        del Z2
        if b_req_grad:
            grad_b = dY.sum(axis=0)
        del dY, grad_output
        return grad_input, grad_W, grad_U, grad_V, grad_b

    @staticmethod
    def backward5_flops(input, W, U, V, b):
        X_req_grad, W_req_grad, U_req_grad, V_req_grad, b_req_grad = \
                input.requires_grad, W.requires_grad, U.requires_grad, \
                V.requires_grad, b.requires_grad if b is not None else None
        nflops = 0
        # Z1 = dY .mm (V.t())
        if U_req_grad:
            nflops += 2 * prod(input.shape[:-1]) * V.shape[1] * V.shape[0]
        # grad_U = X.t() .mm (Z1)
        if U_req_grad:
            nflops += 2 * prod(input.shape) * V.shape[0]
        # Z2 = X .mm (U)
        if V_req_grad:
            nflops += 2 * prod(input.shape) * U.shape[1]
        # grad_V = Z2.t() .mm (dY)
        if V_req_grad:
            nflops += 2 * prod(input.shape[:-1]) * U.shape[1] * W.shape[1]
        # Z3 = W .addmm (U, V)
        if X_req_grad:
            nflops += 2 * U.shape[0] * U.shape[1] * V.shape[1]
        # grad_input = dY .mm (Z3.t())
        if X_req_grad:
            nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops


light_lora_collection = LightLoRACollection()


# class LightLoRA(torch.nn.Linear):
#     def __init__(self, in_features, out_features, rank, bias=True,
#                  device=None, dtype=None, fastest: str = "flops"):
#         if fastest not in ["flops", "benchmark", "benchmark_short"]:
#             raise ValueError("Possible values for \"fastest\" are " \
#                     "\"flops\", \"benchmark\", \"benchmark_short\"")
#         super().__init__(in_features, out_features, bias, device, dtype)
#         self.fastest = fastest
#         self.U = torch.nn.Parameter(torch.randn(in_features, rank,
#                                     device=device, dtype=dtype))
#         self.V = torch.nn.Parameter(torch.randn(rank, out_features,
#                                     device=device, dtype=dtype))
#         self.weight.requires_grad = False
    
#     def forward(self, x):
#         if self.fastest == "flops":
#             path_f, path_b = light_lora_collection.get_best_by_flops(
#                 x, self.weight.t(), self.U, self.V, self.bias)
#         elif self.fastest == "benchmark":
#             path_f, path_b = light_lora_collection.get_best_by_bench(
#                 x, self.weight.t(), self.U, self.V, self.bias)
#         elif self.fastest == "benchmark_short":
#             path_f, path_b = light_lora_collection.get_best_by_bench_short(
#                 x, self.weight.t(), self.U, self.V, self.bias)
#         y = light_lora_collection[path_f, path_b].apply(
#             x, self.weight.t(), self.U, self.V, self.bias)
#         return y

#     def flops(self, x):
#         if self.fastest == "flops":
#             path_f, path_b = light_lora_collection.get_best_by_flops(x, \
#                     self.weight.t(), self.U, self.V, self.bias)
#         elif self.fastest == "benchmark":
#             path_f, path_b = light_lora_collection.get_best_by_bench(x, \
#                     self.weight.t(), self.U, self.V, self.bias)
#         elif self.fastest == "benchmark_short":
#             path_f, path_b = light_lora_collection.get_best_by_bench_short(x, \
#                     self.weight.t(), self.U, self.V, self.bias)
#         nflops = light_lora_collection[path_f, path_b].flops(x, \
#                 self.weight.t(), self.U, self.V, self.bias)
#         return nflops

