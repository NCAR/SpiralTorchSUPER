import torch
from torch import Tensor

__all__ = ["st_fista_subproblem"]

def st_fista_subproblem(b: Tensor, lam1: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    """
    Solves optimization subproblem by using FISTA [1].
    For most MLE applications
    b = x_k - 1/alpha * grad(f)
        where x_k is the current state vector, 1/alpha is the step size,
        f - is the error function so grad(f) is the gradient of f with respect to x

    lam - TV penalty function is typically
        lam = 1/alpha or tau/alpha

    returns x - the solution to the subproblem

    Solving the sub problem should be done separately for each separable variable.
        for example, this function should be run once for backscatter coefficient, lidar ratio and depolarization each.
        For non-TV variables (gain, deadtime), don't use this function call.  Just run steepest descent:
        x_{k+1} = x_k - 1/alpha * grad(x)

    This function expects rectangular arrays.  Mapping non-rectangular spaces (e.g. altitude varying range data)
    needs to happen prior to calling this function.
    """
    return torch.ops.STCuda.st_fista_subproblem.default(b, lam1, lb, ub)

@torch.library.register_fake("STCuda::st_fista_subproblem")
def _(b, lam1, lb, ub):
    torch._check(b.shape == lb.shape)
    torch._check(b.shape == ub.shape)
    torch._check(b.dtype == torch.float)
    torch._check(lb.dtype == torch.float)
    torch._check(ub.dtype == torch.float)
    return torch.empty_like(b)

