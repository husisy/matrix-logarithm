import itertools
import numpy as np
import scipy.linalg
import torch

from utils import TorchMatrixLogm, TorchMatrixSqrtm


def test_matrix_inverse_autodiff():
    N0 = 5
    np_rng = np.random.default_rng()
    np0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0)) + N0*np.eye(N0)
    np1 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))

    torch0 = torch.tensor(np0, dtype=torch.complex128, requires_grad=True)
    torch1 = torch.tensor(np1, dtype=torch.complex128)
    loss = torch.sum(torch.linalg.inv(torch0) * torch1).sum().real
    loss.backward()
    ret_ = torch0.grad.detach().numpy().copy()
    tmp0 = np.linalg.inv(np0)
    ret0 = -tmp0.T.conj() @ np1.conj() @ tmp0.T.conj()
    assert np.abs(ret_-ret0).max() < 1e-7


def random_positive_matrix(N, tag_complex=False, seed=None):
    np_rng = np.random.default_rng(seed)
    if tag_complex:
        np0 = np_rng.normal(size=(N,N)) + 1j*np_rng.normal(size=(N,N))
        ret = np0 @ np0.T.conj()
    else:
        np0 = np_rng.normal(size=(N,N))
        ret = np0 @ np0.T
    ret = ret + 1e-6*np.eye(N)
    return ret


def test_TorchMatrixSqrtm_real_autodiff():
    N0 = 5
    zero_eps = 1e-5
    np_rng = np.random.default_rng()
    np0 = random_positive_matrix(N0, tag_complex=False)
    np1 = np_rng.normal(size=(N0,N0))

    hf0 = lambda x,np1: np.sum(scipy.linalg.sqrtm(x) * np1)
    hf0 = lambda x,np1: np.sum(scipy.linalg.sqrtm(scipy.linalg.sqrtm(x)) * np1)
    ret_ = np.zeros((N0,N0), dtype=np.float64)
    for ind0,ind1 in itertools.product(range(N0),range(N0)):
        tmp0,tmp1 = [np0.copy() for _ in range(2)]
        tmp0[ind0,ind1] += zero_eps
        tmp1[ind0,ind1] -= zero_eps
        ret_[ind0,ind1] = (hf0(tmp0,np1)-hf0(tmp1,np1)) / (2*zero_eps)

    torch0 = torch.tensor(np0, dtype=torch.float64, requires_grad=True)
    torch1 = torch.tensor(np1, dtype=torch.float64)
    loss = torch.sum(TorchMatrixSqrtm.apply(TorchMatrixSqrtm.apply(torch0)) * torch1)
    loss.backward()
    ret0 = torch0.grad.detach().numpy().copy()
    assert np.abs(ret_-ret0).max()<1e-7


def test_TorchMatrixSqrtm_complex_autodiff():
    N0 = 5
    zero_eps = 1e-5
    np_rng = np.random.default_rng()
    np0 = random_positive_matrix(N0, tag_complex=True)
    np1 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))

    hf0 = lambda x,np1: np.sum(scipy.linalg.sqrtm(x) * np1).real
    hf0 = lambda x,np1: np.sum(scipy.linalg.sqrtm(scipy.linalg.sqrtm(x)) * np1).real
    ret_ = np.zeros((N0,N0), dtype=np.complex128)
    for ind0,ind1 in itertools.product(range(N0),range(N0)):
        tmp0,tmp1,tmp2,tmp3 = [np0.copy() for _ in range(4)]
        tmp0[ind0,ind1] += zero_eps
        tmp1[ind0,ind1] -= zero_eps
        tmp2[ind0,ind1] += zero_eps*1j
        tmp3[ind0,ind1] -= zero_eps*1j
        ret_[ind0,ind1] = (hf0(tmp0,np1)-hf0(tmp1,np1)) / (2*zero_eps) + 1j*(hf0(tmp2,np1)-hf0(tmp3,np1)) / (2*zero_eps)

    torch0 = torch.tensor(np0, dtype=torch.complex128, requires_grad=True)
    torch1 = torch.tensor(np1, dtype=torch.complex128)
    loss = torch.sum(TorchMatrixSqrtm.apply(TorchMatrixSqrtm.apply(torch0)) * torch1).real
    loss.backward()
    ret0 = torch0.grad.detach().numpy().copy()
    assert np.abs(ret_-ret0).max()<1e-7


def test_TorchMatrixLogm_real_autodiff():
    np_rng = np.random.default_rng()
    N0 = 8
    np0 = random_positive_matrix(N0, tag_complex=False)
    np1 = np_rng.normal(size=(N0,N0))

    zero_eps = 1e-4
    ret_ = np.zeros((N0,N0), dtype=np.float64)
    hf0 = lambda A,B: np.sum(scipy.linalg.logm(A)*B)
    for ind0,ind1 in itertools.product(range(N0),range(N0)):
        tmp0,tmp1 = [np0.copy() for _ in range(2)]
        tmp0[ind0,ind1] += zero_eps
        tmp1[ind0,ind1] -= zero_eps
        ret_[ind0,ind1] = (hf0(tmp0,np1)-hf0(tmp1,np1)) / (2*zero_eps)

    op_logm = TorchMatrixLogm(num_sqrtm=8, pade_order=8)

    torch0 = torch.tensor(np0, dtype=torch.float64, requires_grad=True)
    torch1 = torch.tensor(np1, dtype=torch.float64)
    loss = torch.sum(op_logm(torch0)*torch1)
    loss.backward()
    ret0 = torch0.grad.detach().numpy().copy()
    assert np.abs(ret_-ret0).max()<1e-5


def test_TorchMatrixLogm_complex_autodiff():
    np_rng = np.random.default_rng()
    N0 = 8
    np0 = random_positive_matrix(N0, tag_complex=True)
    np1 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))

    zero_eps = 1e-4
    ret_ = np.zeros((N0,N0), dtype=np.complex128)
    hf0 = lambda A,B: np.sum(scipy.linalg.logm(A)*B).real
    for ind0,ind1 in itertools.product(range(N0),range(N0)):
        tmp0,tmp1,tmp2,tmp3 = [np0.copy() for _ in range(4)]
        tmp0[ind0,ind1] += zero_eps
        tmp1[ind0,ind1] -= zero_eps
        tmp2[ind0,ind1] += zero_eps*1j
        tmp3[ind0,ind1] -= zero_eps*1j
        ret_[ind0,ind1] = (hf0(tmp0,np1)-hf0(tmp1,np1)) / (2*zero_eps) + 1j*(hf0(tmp2,np1)-hf0(tmp3,np1)) / (2*zero_eps)

    op_logm = TorchMatrixLogm(num_sqrtm=8, pade_order=8)

    torch0 = torch.tensor(np0, dtype=torch.complex128, requires_grad=True)
    torch1 = torch.tensor(np1, dtype=torch.complex128)
    loss = torch.sum(op_logm(torch0)*torch1).real
    loss.backward()
    ret0 = torch0.grad.detach().numpy().copy()
    assert np.abs(ret_-ret0).max()<1e-5
