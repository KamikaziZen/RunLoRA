import torch

from torch.cuda.amp import custom_fwd, custom_bwd
from math import prod

class LightLoRACollection(object):
    def __init__(self):
        self.forward = {1: self.forward1, 2: self.forward2}
        self.forward_flops = {1: self.forward1_flops, 2: self.forward2_flops}
        self.backward = {1: self.backward1, 2: self.backward2, \
                         3: self.backward3, 4: self.backward4, \
                         5: self.backward5}
        self.backward_flops = {1: self.backward1_flops, 2: self.backward2_flops, \
                         3: self.backward3_flops, 4: self.backward4_flops, \
                         5: self.backward5_flops}

    @staticmethod
    @custom_fwd
    def forward1(ctx, input, W, U, V):
        """Y=XW+(XU)V save(X,W,U,V)"""
        ctx.save_for_backward(input, W, U, V)
        return input.mm(W).addmm(input.mm(U), V)
      
    @staticmethod
    def forward1_flops(input, W, U, V):
        nflops = 0
        # input .mm (W)
        nflops += 2 * prod(input.shape) * W.shape[1]
        # input .mm (U)
        nflops += 2 * prod(input.shape) * U.shape[1]
        # (input.mm(W)) .addmm_ (input.mm(U), V)
        nflops += 2 * prod(input.shape[:-1]) * U.shape[1] * V.shape[1]
        return nflops

    @staticmethod
    @custom_fwd
    def forward2(ctx, input, W, U, V):
        """Y=X(W+UV) save(X,W,U,V)"""
        ctx.save_for_backward(input, W, U, V)
        return input.mm(W.addmm(U, V))

    @staticmethod
    def forward2_flops(input, W, U, V):
        nflops = 0
        # W .addmm (U, V)
        nflops += 2 * U.shape[0] * U.shape[1] * V.shape[1]
        # input .mm (W.addmm(U, V))
        nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops

    @staticmethod
    @custom_bwd
    def backward1(ctx, grad_output):
        """load(X,W,U,V) Z=dYV' dU=X'Z dV=(XU)'dY dX=dYW'+ZU'"""
        input, W, U, V = ctx.saved_tensors
        Z = grad_output.mm(V.t())
        grad_U = input.t().mm(Z)
        # If input is a 3D tensor, we need a tensor dot over several axes
        grad_V = (input.mm(U)).t().mm(grad_output)
        grad_input = grad_output.mm(W.t()).addmm(Z, U.t())
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    def backward1_flops(input, W, U, V):
        nflops = 0
        # grad_output .mm (V.t())
        nflops += 2 * prod(input.shape[:-1]) * V.shape[1] * V.shape[0]
        # input.t() .mm (Z)
        nflops += 2 * prod(input.shape) * V.shape[0]
        # input .mm (U)
        nflops += 2 * prod(input.shape) * U.shape[1]
        # (input.mm(U)).t() .mm (grad_output)
        nflops += 2 * U.shape[1] * prod(input.shape[:-1]) * W.shape[1]
        # grad_output .mm (W.t())
        nflops += 2 * prod(input.shape) * W.shape[1]
        # grad_output.mm(W.t()) .addmm_ (Z, U.t())
        nflops += 2 * prod(input.shape) * V.shape[0]
        return nflops

    @staticmethod
    @custom_bwd
    def backward2(ctx, grad_output):
        """load(X,W,U,V) Z=dYV' dU=X'Z dV=U'(X'dY) dX=dYW'+ZU'"""
        input, W, U, V = ctx.saved_tensors
        Z = grad_output.mm(V.t())
        grad_U = input.t().mm(Z)
        # If input is a 3D tensor, we need a reshape here to multiply oveer several axes
        grad_V = U.t().mm(input.t().mm(grad_output))
        grad_input = grad_output.mm(W.t()).addmm_(Z, U.t())
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    def backward2_flops(input, W, U, V):
        nflops = 0
        # grad_output .mm (V.t())
        nflops += 2 * prod(input.shape[:-1]) * V.shape[1] * V.shape[0]
        # input.t().mm (Z)
        nflops += 2 * prod(input.shape) * V.shape[0]
        # input.t() .mm (grad_output)
        nflops += 2 * prod(input.shape) * W.shape[1]
        # U.t() .mm (input.t().mm(grad_output))
        nflops += 2 * U.shape[0] * U.shape[1] * W.shape[1]
        # grad_output .mm (W.t())
        nflops += 2 * prod(input.shape) * W.shape[1]
        # grad_output.mm(W.t()) .addmm_ (Z, U.t())
        nflops += 2 * prod(input.shape[:-1]) * V.shape[0] * U.shape[0]
        return nflops

    @staticmethod
    @custom_bwd
    def backward3(ctx, grad_output):
        """load(X,W,U,V) Z=X'dY dU=ZV' dV=U'Z dX=dYW'+(dYV')U'"""
        input, W, U, V = ctx.saved_tensors
        # If input is a 3D tensor, need to reshape for proper matrix multiplication
        Z = input.t().mm(grad_output)
        grad_U = Z.mm(V.t())
        grad_V = (U.t()).mm(Z)
        grad_input = grad_output.mm(W.t()).addmm_(grad_output.mm(V.t()), U.t())
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    def backward3_flops(input, W, U, V):
        nflops = 0
        # input.t() .mm (grad_output)
        nflops += 2 * prod(input.shape) * W.shape[1]
        # Z .mm (V.t())
        nflops += 2 * input.shape[-1] * V.shape[1] * V.shape[0]
        # (U.t()) .mm (Z)
        nflops += 2 * U.shape[1] * U.shape[0] * W.shape[1]
        # grad_output .mm (W.t())
        nflops += 2 * prod(input.shape) * W.shape[1]
        # grad_output .mm (V.t())
        nflops += 2 * prod(input.shape[:-1]) * V.shape[1] * V.shape[0]
        # grad_output.mm(W.t()) .addmm_ (grad_output.mm(V.t()), U.t())
        nflops += 2 * prod(input.shape[:-1]) * U.shape[1] * U.shape[0]
        return nflops

    @staticmethod
    @custom_bwd
    def backward4(ctx, grad_output):
        """load(X,W,U,V) Z=X'dY dU=ZV' dV=U'Z dX=dY(W+UV)'"""
        input, W, U, V = ctx.saved_tensors
        # If input is a 3D tensor, need to reshape for proper matrix multiplication
        Z = input.t().mm(grad_output)
        grad_U = Z.mm(V.t())
        grad_V = (U.t()).mm(Z)
        grad_input = grad_output.mm((W.addmm(U, V)).t())
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    def backward4_flops(input, W, U, V):
        nflops = 0
        # input.t() .mm (grad_output)
        nflops += 2 * prod(input.shape) * W.shape[1]
        # Z .mm (V.t())
        nflops += 2 * W.shape[0] * V.shape[1] * V.shape[0]
        # (U.t()) .mm (Z)
        nflops += 2 * U.shape[1] * U.shape[0] * W.shape[1]
        # W .addmm (U, V)
        nflops += 2 * U.shape[0] * U.shape[1] * V.shape[1]
        # grad_output .mm ((W.addmm(U, V)).t())
        nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops

    @staticmethod
    @custom_bwd
    def backward5(ctx, grad_output):
        """load(X,W,U,V) dU=X'(dYV') dV=(XU)'dY dX=dY(W+UV)'"""
        input, W, U, V = ctx.saved_tensors
        # If input is a 3D tensor, we need a reshape here
        grad_U = input.t().mm(grad_output.mm(V.t()))
        grad_V = (input.mm(U)).t().mm(grad_output)
        grad_input = grad_output.mm((W.addmm(U, V)).t())
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    def backward5_flops(input, W, U, V):
        nflops = 0
        # grad_output .mm (V.t())
        nflops += 2 * prod(input.shape[:-1]) * V.shape[1] * V.shape[0]
        # input.t() .mm (grad_output.mm(V.t()))
        nflops += 2 * prod(input.shape) * V.shape[0]
        # input .mm (U)
        nflops += 2 * prod(input.shape) * U.shape[1]
        # (input.mm(U)).t() .mm (grad_output)
        nflops += 2 * prod(input.shape[:-1]) * U.shape[1] * W.shape[1]
        # W .addmm (U, V)
        nflops += 2 * U.shape[0] * U.shape[1] * V.shape[1]
        # grad_output .mm ((W.addmm(U, V)).t())
        nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops

    def __call__(self, path_f, path_b):
        class LightLoRA(torch.autograd.Function):
            forward = self.forward[path_f]
            backward = self.backward[path_b]
            def flops(input, W, U, V):
                return self.forward_flops[path_f](input, W, U, V) \
                    + self.backward_flops[path_b](input, W, U, V)
        return LightLoRA

class LightLoRACollection_wpuv(object):
    def __init__(self):
        self.forward = {1: self.forward1}
        self.backward = {1: self.backward1, 2: self.backward2}

    @staticmethod
    @custom_fwd
    def forward1(ctx, input, W, U, V):
        """Y=X(W+UV) save(X,W,U,V,W+UV)"""
        Z = W.addmm(U, V)
        ctx.save_for_backward(input, W, U, V, Z)
        return input.mm(Z)

    @staticmethod
    @custom_bwd
    def backward1(ctx, grad_output):
        """load(X,W,U,V,WpUV) Z=X'dY dU=ZV' dV=U'Z dX=dY(W+UV)'"""
        input, W, U, V, WpUV = ctx.saved_tensors
        Z = input.t().mm(grad_output)
        grad_U = Z.mm(V.t())
        grad_V = (U.t()).mm(Z)
        grad_input = grad_output.mm(WpUV.t())
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2(ctx, grad_output):
        """load(X,W,U,V,WpUV) dU=X'(dYV') dV=(XU)'dY dX=dY(W+UV)'"""
        input, W, U, V, WpUV = ctx.saved_tensors
        grad_U = input.t().mm(grad_output.mm(V.t()))
        grad_V = (input.mm(U)).t().mm(grad_output)
        grad_input = grad_output.mm(WpUV.t())
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    def __call__(self, path_f, path_b):
        class LightLoRA_wpuv(torch.autograd.Function):
            forward = self.forward[path_f]
            backward = self.backward[path_b]
        return LightLoRA_wpuv

class LightLoRACollection_xu(object):
    def __init__(self):
        self.forward = {1: self.forward1}
        self.backward = {1: self.backward1}

    @staticmethod
    @custom_fwd
    def forward1(ctx, input, W, U, V):
        """Y=XW+(XU)V save(X,W,U,V,XU)"""
        Z = input.mm(U)
        ctx.save_for_backward(input, W, U, V, Z)
        return input.mm(W).addmm_(input.mm(U), V)

    @staticmethod
    @custom_bwd
    def backward1(ctx, grad_output):
        """load(X,W,U,V,XU) Z=dYV' dU=X'Z dV=(XU)'dY dX=dYW'+ZU'"""
        input, W, U, V, XU = ctx.saved_tensors
        Z = grad_output.mm(V.t())
        grad_U = input.t().mm(Z)
        grad_V = XU.t().mm(grad_output)
        grad_input = grad_output.mm(W.t()).addmm_(Z, U.t())
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    def __call__(self, path_f, path_b):
        class LightLoRA_xu(torch.autograd.Function):
            forward = self.forward[path_f]
            backward = self.backward[path_b]
        return LightLoRA_xu

