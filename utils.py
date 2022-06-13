import torch
import numpy as np
import scipy.linalg

class TorchMatrixSqrtm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matA):
        ret = torch.from_numpy(scipy.linalg.sqrtm(matA.numpy()))
        ctx.save_for_backward(ret)
        return ret
    @staticmethod
    def backward(ctx, grad_output):
        matB, = ctx.saved_tensors
        matB_np = matB.numpy()
        gradA_np = scipy.linalg.solve_sylvester(matB_np, matB_np, grad_output.numpy())
        gradA = torch.from_numpy(gradA_np)
        return gradA,


class PartMatrixLogarithm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matA, s):
        matA_np = matA.numpy()
        for _ in range(s):
            matA_np = scipy.linalg.sqrtm(matA_np)
        matB = torch.from_numpy(matA_np)
        ctx.save_for_backward(matB, torch.tensor(s))
        return matB
    @staticmethod
    def backward(ctx, grad_output):
        matB,s = ctx.saved_tensors
        matB_np = matB.numpy()
        gradA = grad_output
        for _ in range(s.item()):
            gradA = scipy.linalg.solve_sylvester(matB_np, matB_np, gradA)
            matB_np = matB_np @ matB_np
        gradA = torch.tensor(gradA)
        return gradA,None


class TorchMatrixLogm(torch.nn.Module):
    def __init__(self, num_sqrtm=8, pade_order=8):
        super().__init__()
        node,weight = np.polynomial.legendre.leggauss(pade_order)
        self.alpha = torch.tensor(weight * 2**(num_sqrtm-1)).view(-1, 1, 1)
        self.beta = torch.tensor((node + 1) / 2).view(-1, 1, 1)
        self.num_sqrtm = num_sqrtm

    def forward(self, matA):
        torch1 = PartMatrixLogarithm.apply(matA, self.num_sqrtm)
        eye0 = torch.eye(matA.shape[0])
        # ret = sum(torch.linalg.solve((1-b)*eye0+b*torch1, a*torch1-a*eye0) for a,b in zip(self.alpha,self.beta))
        tmp0 = (1-self.beta)*eye0+self.beta*torch1
        tmp1 = self.alpha*torch1 - self.alpha*eye0
        ret = torch.linalg.solve(tmp0, tmp1).sum(dim=0)
        return ret
