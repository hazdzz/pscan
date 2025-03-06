# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/
# Written by Francois Fleuret <francois@fleuret.org>


import os
import random
import torch


class PScan(torch.autograd.Function):
    # A: (b, n)
    # X: (b, n, d)
    # Y_init: (b, d)
    # Y: (b, n, d)
    #
    # Y[:, t] = A[:, 0] * Y_init    + X[:, 0] if t == 0
    #         = A[:, t] * Y[:, t-1] + X[:, t] otherwise

    @staticmethod
    def expand_(A, X):
        if A.size(1) == 1:
            return
        T = 2 * (A.size(1) // 2)
        Aa = A[:, :T].view(A.size(0), T//2, 2, -1)
        Xa = X[:, :T].view(X.size(0), T//2, 2, -1)
        Xa[:, :, 1].add_(Aa[:, :, 1] * Xa[:, :, 0])
        Aa[:, :, 1].mul_(Aa[:, :, 0])
        PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])
        Xa[:, 1:, 0].add_(Aa[:, 1:, 0] * Xa[:, :-1, 1])
        Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
        if T < A.size(1):
            X[:, -1].add_(A[:, -1] * X[:, -2])
            A[:, -1].mul_(A[:, -2])


    @staticmethod
    def acc_rev_(A, X):
        if X.size(1) == 1:
            return
        T = 2 * (X.size(1) // 2)
        Aa = A[:, -T:].view(A.size(0), T//2, 2, -1)
        Xa = X[:, -T:].view(X.size(0), T//2, 2, -1)
        Xa[:, :, 0].add_(Aa[:, :, 1].conj() * Xa[:, :, 1])
        B = Aa[:, :, 0].clone()
        B[:, 1:].mul_(Aa[:, :-1, 1].conj())
        PScan.acc_rev_(B, Xa[:, :, 0])
        Xa[:, :-1, 1].add_(Aa[:, 1:, 0].conj() * Xa[:, 1:, 0])
        if T < A.size(1):
            X[:, 0].add_(A[:, 1].conj() * X[:, 1])


    @staticmethod
    def forward(ctx, A, X, Y_init):
        ctx.A = A[:, :, None].clone()
        ctx.Y_init = Y_init[:, None, :].clone()
        ctx.A_star = ctx.A.clone()
        ctx.X_star = X.clone()
        PScan.expand_(ctx.A_star, ctx.X_star)
        return ctx.A_star * ctx.Y_init + ctx.X_star


    @staticmethod
    def backward(ctx, grad_output):
        U = grad_output * ctx.A_star.conj()
        A = ctx.A.clone()
        R = grad_output.clone()
        PScan.acc_rev_(A, R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1].conj()).add_(ctx.X_star[:, :-1])
        grad_A = (Q.conj() * R).sum(-1)
        return grad_A, R, U.sum(dim=1)


pscan_len = PScan.apply


def pscan_feat(A, X, Y_init):
    A_ = A.unsqueeze(-1)          # (b, d) -> (b, d, 1)
    X_ = X.permute(0, 2, 1)       # (b, n, d) -> (b, d, n)
    
    # (b, d, num_chunks, n)
    Y_ = pscan_len(A_, X_, Y_init)
    
    if Y_.dim() == 4:
        Y_ = Y_.reshape(Y_.size(0), -1, Y_.size(-1))  # (b, d * num_chunks, n)
        Y_ = Y_[:, :A.size(1), :]
    
    # (b, n, d)
    return Y_.permute(0, 2, 1)


def set_env(seed = 42) -> None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    device = torch.device('cuda')
    set_env(42)

    b, n, d = 1, 6, 4

    x = torch.rand(b, n, d) + 1j * torch.rand(1, 6, 4)
    p = torch.rand(b, n-1)
    a = torch.zeros(b, n)
    h = torch.zeros_like(x)

    a[:, 1:] = p
    h[:, 0, :] = x[:, 0, :]
    h_init = h[:, 0, :].clone()
    for k in range(1, n):
        h[:, k, :] = p[:, k-1].unsqueeze(-1) * h[:, k-1, :] + x[:, k, :]
    h_ = pscan_len(a, x, h_init)

    print(torch.allclose(h, h_, rtol=1e-5, atol=1e-6))

    x = torch.rand(b, n, d) + 1j * torch.rand(1, 6, 4)
    q = torch.rand(b, d-1)
    b = torch.zeros(b, d)
    i = torch.zeros_like(x)

    b[:, 1:] = q
    i[:, :, 0] = x[:, :, 0]
    i_init = i[:, :, 0].clone()
    for k in range(1, d):
        i[:, :, k] = q[:, k-1].unsqueeze(1) * i[:, :, k-1] + x[:, :, k]
    i_ = pscan_feat(b, x, i_init)

    print(torch.allclose(i, i_, rtol=1e-5, atol=1e-6))