import torch

from torch.cuda.amp import custom_fwd, custom_bwd
from math import prod

class LightLoRACollection(object):
    def __init__(self):
        self.forward_keys = [i for i in dir(self) \
                if i.find("forward") != -1 and i[-5:] != "flops"]
        self.backward_keys = [i for i in dir(self) \
                if i.find("backward") != -1 and i[-5:] != "flops"]

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
            def flops(input, W, U, V):
                return method_forward_flops(input, W, U, V) \
                    + method_backward_flops(input, W, U, V)
        return LightLoRA

    @staticmethod
    @custom_fwd
    def forward1(ctx, input, W, U, V):
        """Y=XW+(XU)V save(X,W,U,V)"""
        X = input.reshape(-1, input.shape[-1])
        Y_shape = torch.Size(list(input.shape[:-1]) + [W.shape[1]])
        ctx.save_for_backward(input, W, U, V)
        return (X.mm(W).addmm_(X.mm(U), V)).reshape(Y_shape)
      
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
        X = input.reshape(-1, input.shape[-1])
        Y_shape = torch.Size(list(input.shape[:-1]) + [W.shape[1]])
        return X.mm(W.addmm(U, V)).reshape(Y_shape)

    @staticmethod
    def forward2_flops(input, W, U, V):
        nflops = 0
        # W .addmm (U, V)
        nflops += 2 * U.shape[0] * U.shape[1] * V.shape[1]
        # input .mm (W.addmm(U, V))
        nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops

    @staticmethod
    @custom_fwd
    def forward3(ctx, input, W, U, V):
        """Y=(XU)V+XW save(X,W,U,V)"""
        X = input.reshape(-1, input.shape[-1])
        Y_shape = torch.Size(list(input.shape[:-1]) + [W.shape[1]])
        ctx.save_for_backward(input, W, U, V)
        return (X.mm(U).mm(V)).addmm_(X, W).reshape(Y_shape)
      
    @staticmethod
    def forward3_flops(input, W, U, V):
        nflops = 0
        # input .mm (W)
        nflops += 2 * prod(input.shape) * W.shape[1]
        # input .mm (U)
        nflops += 2 * prod(input.shape) * U.shape[1]
        # (input.mm(W)) .addmm_ (input.mm(U), V)
        nflops += 2 * prod(input.shape[:-1]) * U.shape[1] * V.shape[1]
        return nflops

    @staticmethod
    @custom_bwd
    def backward1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU dU=X'Z dV=Z2'dY dX=dYW'+ZU'"""
        input, W, U, V = ctx.saved_tensors
        X = input.reshape(-1, input.shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        grad_V = (X.mm(U)).t().mm(dY)
        grad_input = dY.mm(W.t()).addmm_(Z1, U.t()).reshape(input.shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward1_X_Z1_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU dU=X'Z dV=Z2'dY dX=dYW'+ZU'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.mm(U)
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input = Z1.mm(U.t())
        del Z1
        grad_V = Z2.t().mm(dY)
        del Z2
        grad_input.addmm_(dY, W.t())
        del dY, grad_output
        grad_input = grad_input.reshape(X_shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward1_X_Z2_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU dU=X'Z dV=Z2'dY dX=dYW'+ZU'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.mm(U)
        grad_U = X.t().mm(Z1)
        del X, input
        grad_V = Z2.t().mm(dY)
        del Z2
        grad_input = Z1.mm(U.t())
        del Z1
        grad_input.addmm_(dY, W.t())
        del dY, grad_output
        grad_input = grad_input.reshape(X_shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward1_X_Z2_dY_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU dU=X'Z dV=Z2'dY dX=dYW'+ZU'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.mm(U)
        grad_U = X.t().mm(Z1)
        del X, input
        grad_V = Z2.t().mm(dY)
        del Z2
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_input.addmm_(Z1, U.t())
        del Z1
        grad_input = grad_input.reshape(X_shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward1_Z1_X_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU dU=X'Z dV=Z2'dY dX=dYW'+ZU'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        grad_input = Z1.mm(U.t())
        del Z1
        Z2 = X.mm(U)
        del X, input
        grad_V = Z2.t().mm(dY)
        del Z2
        grad_input.addmm_(dY, W.t())
        del dY, grad_output
        grad_input = grad_input.reshape(X_shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward1_Z2_X_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU dU=X'Z dV=Z2'dY dX=dYW'+ZU'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.mm(U)
        grad_V = Z2.t().mm(dY)
        del Z2
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input = Z1.mm(U.t())
        del Z1
        grad_input.addmm_(dY, W.t())
        del dY, grad_output
        grad_input = grad_input.reshape(X_shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward1_Z2_X_dY_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU dU=X'Z dV=Z2'dY dX=dYW'+ZU'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.mm(U)
        grad_V = Z2.t().mm(dY)
        del Z2
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_input.addmm_(Z1, U.t())
        del Z1
        grad_input = grad_input.reshape(X_shape)
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
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X = input.reshape(-1, input.shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        grad_V = U.t().mm(X.t().mm(dY))
        grad_input = dY.mm(W.t()).addmm_(Z1, U.t()).reshape(input.shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_X_dY_Z1_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.t().mm(dY)
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        grad_V = U.t().mm(Z2)
        del Z2
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_X_dY_Z2_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.t().mm(dY)
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_V = U.t().mm(Z2)
        del Z2
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_X_Z1_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.t().mm(dY)
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input = Z1.mm(U.t())
        del Z1
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
        grad_V = U.t().mm(Z2)
        del Z2
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_X_Z1_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.t().mm(dY)
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input = Z1.mm(U.t())
        del Z1
        grad_V = U.t().mm(Z2)
        del Z2
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_X_Z2_dY_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.t().mm(dY)
        grad_U = X.t().mm(Z1)
        del X, input
        grad_V = U.t().mm(Z2)
        del Z2
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_X_Z2_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.t().mm(dY)
        grad_U = X.t().mm(Z1)
        del X, input
        grad_V = U.t().mm(Z2)
        del Z2
        grad_input = Z1.mm(U.t())
        del Z1
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_dY_X_Z1_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.t().mm(dY)
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        grad_V = U.t().mm(Z2)
        del Z2
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_dY_X_Z2_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.t().mm(dY)
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_U = X.t().mm(Z1)
        del X, input
        grad_V = U.t().mm(Z2)
        del Z2
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_dY_Z2_X_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        Z2 = X.t().mm(dY)
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_V = U.t().mm(Z2)
        del Z2
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_Z1_X_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        grad_input = Z1.mm(U.t())
        del Z1
        Z2 = X.t().mm(dY)
        del X, input
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
        grad_V = U.t().mm(Z2)
        del Z2
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_Z1_X_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        grad_input = Z1.mm(U.t())
        del Z1
        Z2 = X.t().mm(dY)
        del X, input
        grad_V = U.t().mm(Z2)
        del Z2
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_Z2_X_dY_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        grad_V = U.t().mm(Z2)
        del Z2
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_Z2_X_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        grad_V = U.t().mm(Z2)
        del Z2
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input = Z1.mm(U.t())
        del Z1
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward2_Z2_dY_X_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=X'Z1 dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        grad_V = U.t().mm(Z2)
        del Z2
        Z1 = dY.mm(V.t())
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_U = X.t().mm(Z1)
        del X, input
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
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
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X = input.reshape(-1, input.shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z = X.t().mm(dY)
        grad_U = Z.mm(V.t())
        grad_V = (U.t()).mm(Z)
        grad_input = dY.mm(W.t()).addmm_(dY.mm(V.t()), U.t()).reshape(input.shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward3_X_dY_Z1_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        del X, input
        Z1 = dY.mm(V.t())
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        grad_U = Z2.mm(V.t())
        grad_V = U.t().mm(Z2)
        del Z2
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward3_X_dY_Z2_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        del X, input
        Z1 = dY.mm(V.t())
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_U = Z2.mm(V.t())
        grad_V = U.t().mm(Z2)
        del Z2
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward3_X_Z1_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        del X, input
        Z1 = dY.mm(V.t())
        grad_input = Z1.mm(U.t())
        del Z1
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
        grad_U = Z2.mm(V.t())
        grad_V = U.t().mm(Z2)
        del Z2
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward3_X_Z1_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        del X, input
        Z1 = dY.mm(V.t())
        grad_input = Z1.mm(U.t())
        del Z1
        grad_U = Z2.mm(V.t())
        grad_V = U.t().mm(Z2)
        del Z2
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward3_X_Z2_dY_Z1(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        del X, input
        grad_U = Z2.mm(V.t())
        grad_V = U.t().mm(Z2)
        del Z2
        Z1 = dY.mm(V.t())
        grad_input = dY.mm(W.t())
        del dY, grad_output
        grad_input.addmm_(Z1, U.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward3_X_Z2_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        del X, input
        grad_U = Z2.mm(V.t())
        grad_V = U.t().mm(Z2)
        del Z2
        Z1 = dY.mm(V.t())
        grad_input = Z1.mm(U.t())
        del Z1
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward3_Z1_X_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        grad_input = Z1.mm(U.t())
        del Z1
        Z2 = X.t().mm(dY)
        del X, input
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
        grad_U = Z2.mm(V.t())
        grad_V = U.t().mm(Z2)
        del Z2
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward3_Z1_X_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYW'+Z1U'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        grad_input = Z1.mm(U.t())
        del Z1
        Z2 = X.t().mm(dY)
        del X, input
        grad_U = Z2.mm(V.t())
        grad_V = U.t().mm(Z2)
        del Z2
        grad_input.addmm_(dY, W.t())
        grad_input = grad_input.reshape(X_shape)
        del dY, grad_output
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
        """load(X,W,U,V) Z1=W+UV Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYZ1'"""
        input, W, U, V = ctx.saved_tensors
        X = input.reshape(-1, input.shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z = X.t().mm(dY)
        grad_U = Z.mm(V.t())
        grad_V = (U.t()).mm(Z)
        grad_input = dY.mm((W.addmm(U, V)).t()).reshape(input.shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward4_X_Z1_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=W+UV Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYZ1'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        del X, input
        Z1 = W.addmm(U, V)
        grad_input = dY.mm(Z1.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        del dY, grad_output
        grad_U = Z2.mm(V.t())
        grad_V = (U.t()).mm(Z2)
        del Z2
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward4_X_Z2_Z1_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=W+UV Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYZ1'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z2 = X.t().mm(dY)
        del X, input
        grad_U = Z2.mm(V.t())
        grad_V = (U.t()).mm(Z2)
        del Z2
        Z1 = W.addmm(U, V)
        grad_input = dY.mm(Z1.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        del dY, grad_output
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward4_Z1_X_dY_Z2(ctx, grad_output):
        """load(X,W,U,V) Z1=W+UV Z2=X'dY dU=Z2V' dV=U'Z2 dX=dYZ1'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = W.addmm(U, V)
        grad_input = dY.mm(Z1.t())
        grad_input = grad_input.reshape(X_shape)
        del Z1
        Z2 = X.t().mm(dY)
        del X, input
        del dY, grad_output
        grad_U = Z2.mm(V.t())
        grad_V = (U.t()).mm(Z2)
        del Z2
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
        """load(X,W,U,V) Z1=dYV' Z2=XU Z3=W+UV dU=X'Z1 dV=Z2'dY dX=dYZ3'"""
        input, W, U, V = ctx.saved_tensors
        X = input.reshape(-1, input.shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        grad_U = X.t().mm(dY.mm(V.t()))
        grad_V = (X.mm(U)).t().mm(dY)
        grad_input = dY.mm((W.addmm(U, V)).t()).reshape(input.shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward5_Z1_X_Z2_Z3_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU Z3=W+UV dU=X'Z1 dV=Z2'dY dX=dYZ3'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        del Z1
        Z2 = X.mm(U)
        del X, input
        grad_V = Z2.t().mm(dY)
        del Z2
        Z3 = W.addmm(U, V)
        grad_input = dY.mm(Z3.t())
        grad_input = grad_input.reshape(X_shape)
        del Z3
        del dY, grad_output
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward5_Z1_X_Z3_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU Z3=W+UV dU=X'Z1 dV=Z2'dY dX=dYZ3'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        del Z1
        Z2 = X.mm(U)
        del X, input
        Z3 = W.addmm(U, V)
        grad_input = dY.mm(Z3.t())
        grad_input = grad_input.reshape(X_shape)
        del Z3
        grad_V = Z2.t().mm(dY)
        del Z2
        del dY, grad_output
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward5_Z1_Z3_X_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU Z3=W+UV dU=X'Z1 dV=Z2'dY dX=dYZ3'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        del Z1
        Z3 = W.addmm(U, V)
        grad_input = dY.mm(Z3.t())
        grad_input = grad_input.reshape(X_shape)
        del Z3
        Z2 = X.mm(U)
        del X, input
        grad_V = Z2.t().mm(dY)
        del Z2
        del dY, grad_output
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    @custom_bwd
    def backward5_Z3_Z1_X_Z2_dY(ctx, grad_output):
        """load(X,W,U,V) Z1=dYV' Z2=XU Z3=W+UV dU=X'Z1 dV=Z2'dY dX=dYZ3'"""
        input, W, U, V = ctx.saved_tensors
        X_shape = input.shape
        X = input.reshape(-1, X_shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z3 = W.addmm(U, V)
        grad_input = dY.mm(Z3.t())
        grad_input = grad_input.reshape(X_shape)
        del Z3
        Z1 = dY.mm(V.t())
        grad_U = X.t().mm(Z1)
        del Z1
        Z2 = X.mm(U)
        del X, input
        grad_V = Z2.t().mm(dY)
        del Z2
        del dY, grad_output
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

class LightLoRACollection_wpuv(object):
    def __init__(self):
        self.forward = {1: self.forward1}
        self.forward_flops = {1: self.forward1_flops}
        self.backward = {1: self.backward1, 2: self.backward2}
        self.backward_flops = {1: self.backward1_flops, 2: self.backward2_flops}

    @staticmethod
    @custom_fwd
    def forward1(ctx, input, W, U, V):
        """Y=X(W+UV) save(X,W,U,V,W+UV)"""
        X = input.reshape(-1, input.shape[-1])
        Y_shape = torch.Size(list(input.shape[:-1]) + [W.shape[1]])
        Z = W.addmm(U, V)
        ctx.save_for_backward(input, W, U, V, Z)
        return X.mm(Z).reshape(Y_shape)

    @staticmethod
    def forward1_flops(input, W, U, V):
        nflops = 0
        # W .addmm (U, V)
        nflops += 2 * U.shape[0] * U.shape[1] * V.shape[1]
        # input .mm (Z)
        nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops

    @staticmethod
    @custom_bwd
    def backward1(ctx, grad_output):
        """load(X,W,U,V,WpUV) Z=X'dY dU=ZV' dV=U'Z dX=dY(W+UV)'"""
        input, W, U, V, WpUV = ctx.saved_tensors
        X = input.reshape(-1, input.shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z = X.t().mm(dY)
        grad_U = Z.mm(V.t())
        grad_V = (U.t()).mm(Z)
        grad_input = dY.mm(WpUV.t()).reshape(input.shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    def backward1_flops(input, W, U, V):
        nflops = 0
        # input.t() .mm (grad_output)
        nflops += 2 * prod(input.shape) * W.shape[1]
        # Z .mm (V.t())
        nflops += 2 * W.shape[0] * V.shape[1] * V.shape[0]
        # (U.t()) .mm (Z)
        nflops += 2 * U.shape[1] * U.shape[0] * W.shape[1]
        # grad_output .mm ((W.addmm(U, V)).t())
        nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops

    @staticmethod
    @custom_bwd
    def backward2(ctx, grad_output):
        """load(X,W,U,V,WpUV) dU=X'(dYV') dV=(XU)'dY dX=dY(W+UV)'"""
        input, W, U, V, WpUV = ctx.saved_tensors
        X = input.reshape(-1, input.shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        grad_U = X.t().mm(dY.mm(V.t()))
        grad_V = (X.mm(U)).t().mm(dY)
        grad_input = dY.mm(WpUV.t()).reshape(input.shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    def backward2_flops(input, W, U, V):
        nflops = 0
        # grad_output .mm (V.t())
        nflops += 2 * prod(input.shape[:-1]) * V.shape[1] * V.shape[0]
        # input.t() .mm (grad_output.mm(V.t()))
        nflops += 2 * prod(input.shape) * V.shape[0]
        # input .mm (U)
        nflops += 2 * prod(input.shape) * U.shape[1]
        # (input.mm(U)).t() .mm (grad_output)
        nflops += 2 * prod(input.shape[:-1]) * U.shape[1] * W.shape[1]
        # grad_output .mm ((W.addmm(U, V)).t())
        nflops += 2 * prod(input.shape) * W.shape[1]
        return nflops

    def __call__(self, path_f, path_b):
        class LightLoRA_wpuv(torch.autograd.Function):
            forward = self.forward[path_f]
            backward = self.backward[path_b]
            def flops(input, W, U, V):
                return self.forward_flops[path_f](input, W, U, V) \
                    + self.backward_flops[path_b](input, W, U, V)
        return LightLoRA_wpuv

class LightLoRACollection_xu(object):
    def __init__(self):
        self.forward = {1: self.forward1}
        self.forward_flops = {1: self.forward1_flops}
        self.backward = {1: self.backward1}
        self.backward_flops = {1: self.backward1_flops}

    @staticmethod
    @custom_fwd
    def forward1(ctx, input, W, U, V):
        """Y=XW+(XU)V save(X,W,U,V,XU)"""
        X = input.reshape(-1, input.shape[-1])
        Y_shape = torch.Size(list(input.shape[:-1]) + [W.shape[1]])
        Z = X.mm(U)
        ctx.save_for_backward(input, W, U, V, Z)
        return X.mm(W).addmm_(Z, V).reshape(Y_shape)

    @staticmethod
    def forward1_flops(input, W, U, V):
        nflops = 0
        # input .mm (U)
        nflops += 2 * prod(input.shape) * U.shape[1]
        # input .mm (W)
        nflops += 2 * prod(input.shape) * W.shape[1]
        # (input.mm(W)) .addmm_ (input.mm(U), V)
        nflops += 2 * prod(input.shape[:-1]) * U.shape[1] * V.shape[1]
        return nflops

    @staticmethod
    @custom_bwd
    def backward1(ctx, grad_output):
        """load(X,W,U,V,XU) Z=dYV' dU=X'Z dV=(XU)'dY dX=dYW'+ZU'"""
        input, W, U, V, XU = ctx.saved_tensors
        X = input.reshape(-1, input.shape[-1])
        dY = grad_output.reshape(-1, grad_output.shape[-1])
        Z = dY.mm(V.t())
        grad_U = X.t().mm(Z)
        grad_V = XU.t().mm(dY)
        grad_input = dY.mm(W.t()).addmm_(Z, U.t()).reshape(input.shape)
        grad_W = None
        return grad_input, grad_W, grad_U, grad_V

    @staticmethod
    def backward1_flops(input, W, U, V):
        nflops = 0
        # grad_output .mm (V.t())
        nflops += 2 * prod(input.shape[:-1]) * V.shape[1] * V.shape[0]
        # input.t() .mm (Z)
        nflops += 2 * prod(input.shape) * V.shape[0]
        # XU.t() .mm (grad_output)
        nflops += 2 * U.shape[1] * prod(input.shape[:-1]) * W.shape[1]
        # grad_output .mm (W.t())
        nflops += 2 * prod(input.shape) * W.shape[1]
        # grad_output.mm(W.t()) .addmm_ (Z, U.t())
        nflops += 2 * prod(input.shape) * V.shape[0]
        return nflops

    def __call__(self, path_f, path_b):
        class LightLoRA_xu(torch.autograd.Function):
            forward = self.forward[path_f]
            backward = self.backward[path_b]
            def flops(input, W, U, V):
                return self.forward_flops[path_f](input, W, U, V) \
                    + self.backward_flops[path_b](input, W, U, V)
        return LightLoRA_xu

