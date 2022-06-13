import numpy as np
import scipy.linalg

import sympy
import sympy.abc
import matplotlib.pyplot as plt
plt.ion()


hf_at_least_nonzero = lambda x,eps=1e-9: ((2*(x>0)-1) * np.maximum(np.abs(x), eps))
def hf_at_least_non_x(x, shift=0, eps=1e-9):
    tmp0 = x - shift
    tmp1 = ((2*(tmp0>0)-1) * np.maximum(np.abs(tmp0), eps))
    ret = tmp1 + shift
    return ret


def pade_approximation(taylor_coeff, p_order):
    # from the lowest order to highest order
    # taylor(np,float,N)
    # (ret0)p(np,float,p_order+1)
    # (ret1)q(np,float,q_order+1) p_order + q_order + 1 = N
    # wiki: https://en.wikipedia.org/wiki/Pad%C3%A9_approximant
    # http://mathworld.wolfram.com/PadeApproximant.html
    # http://mathfaculty.fullerton.edu/mathews/n2003/pade/PadeApproximationProof.pdf
    len_taylor = len(taylor_coeff)
    #len_p = p_order + 1
    q_order = len_taylor - 1 - p_order #len_q = q_order + 1
    assert (p_order>=0) and (q_order>=0)

    if q_order==0:
        pcoeff = taylor_coeff.copy()
        qcoeff = np.array([1])
    else:
        tmp0 = -taylor_coeff[:-1]
        tmp1 = [-taylor_coeff[0]] + [0]*(q_order-1)
        tmp2 = scipy.linalg.toeplitz(tmp0, tmp1)
        matA = np.concatenate([np.eye(p_order+q_order, p_order), tmp2], axis=1)
        matb = taylor_coeff[1:]
        tmp0 = np.linalg.solve(matA, matb)
        pcoeff = np.concatenate([[taylor_coeff[0]], tmp0[:p_order]])
        qcoeff = np.concatenate([[1],tmp0[p_order:]])
    return pcoeff, qcoeff


def sympy_taylor_coeff(expr, x, x0, order):
    z0 = sympy.series(expr, x, x0, n=order)
    ret = np.array([float(z0.coeff(x, i).evalf()) for i in range(order)])
    return ret


def demo_pade_approximation_sin_over_x():
    taylor_order = 13
    hf0 = lambda x: np.sin(6*x) / hf_at_least_nonzero(x)
    expr0 = sympy.sin(6*sympy.abc.x) / sympy.abc.x
    taylor_coeff = sympy_taylor_coeff(expr0, sympy.abc.x, 0, taylor_order)

    p_order_list = [0,4,6,8,12]
    pcoeff_list = []
    qcoeff_list = []
    for p_i in p_order_list:
        tmp0 = pade_approximation(taylor_coeff, p_order=p_i)
        pcoeff_list.append(tmp0[0])
        qcoeff_list.append(tmp0[1])

    xdata = np.linspace(-1.5, 1.5, 100)
    fig,ax = plt.subplots()
    ax.plot(xdata, hf0(xdata), '.')
    for p_i,pcoeff,qcoeff in zip(p_order_list,pcoeff_list,qcoeff_list):
        poly_p = np.polynomial.Polynomial(pcoeff)
        poly_q = np.polynomial.Polynomial(qcoeff)
        ypade = poly_p(xdata) / poly_q(xdata)
        ax.plot(xdata, ypade, label=f'pade({p_i}/{len(qcoeff)-1})')
    ax.legend()
    ax.set_ylim(-2, 8)
    ax.set_title('sin(x)/x')
    fig.savefig('data/pade_sin_over_x.png')


def demo_pade_approximation_log1p():
    # log(1-x)= \sum_n{x^n/n}
    taylor_order = 13
    hf0 = lambda x: np.log(1+x)
    taylor_coeff = np.concatenate([[0], (-1)**np.arange(taylor_order-1)/np.arange(1,taylor_order)])

    p_order_list = [1,4,6,8,12]
    #p_order=4,6,8 are all good, p_order=0 will diverge, p_order=1,12 is totally bad
    pcoeff_list = []
    qcoeff_list = []
    for p_i in p_order_list:
        tmp0 = pade_approximation(taylor_coeff, p_order=p_i)
        pcoeff_list.append(tmp0[0])
        qcoeff_list.append(tmp0[1])

    xdata = np.linspace(-0.5, 3, 100)
    fig,ax = plt.subplots()
    ax.plot(xdata, hf0(xdata), '.')
    for p_i,pcoeff,qcoeff in zip(p_order_list,pcoeff_list,qcoeff_list):
        poly_p = np.polynomial.Polynomial(pcoeff)
        poly_q = np.polynomial.Polynomial(qcoeff)
        ypade = poly_p(xdata) / poly_q(xdata)
        ax.plot(xdata, ypade, label=f'pade({p_i}/{len(qcoeff)-1})')
    ax.legend()
    ax.set_ylim(-1, 2)
    ax.set_title('log(1+x), pade')
    fig.savefig('data/pade_log1p.png')


def fractional_pade_approximation_log(order):
    node,weight = np.polynomial.legendre.leggauss(order)
    alpha = weight / 2
    beta = (node + 1) / 2
    return alpha,beta


def demo_fractional_pade_approximation_log():
    fig, ax = plt.subplots()
    xdata = np.linspace(-1+1e-4, 2, 100)
    ax.plot(xdata, np.log(hf_at_least_non_x(1+xdata)), '.')
    for order in [4,8,12,16,32]:
        # order=32 (0.001)
        alpha,beta = fractional_pade_approximation_log(order)
        ypade = (alpha[:,np.newaxis] * xdata / (1+beta[:,np.newaxis]*xdata)).sum(axis=0)
        ax.plot(xdata, ypade, label=f'pade({order})')
    ax.legend()
    # ax.set_ylim(-1, 2)
    ax.set_title('log(1+x), fractional pade')
    fig.savefig('data/pade_log1p_fractional.png')

    fig, ax = plt.subplots()
    xdata = np.linspace(-1-1e-3, 0, 100)
    ax.plot(xdata, (xdata+1)*np.log(hf_at_least_non_x(1+xdata)), '.')
    for order in [4,8,12,16,32]:
        # order=32 (0.001)
        alpha,beta = fractional_pade_approximation_log(order)
        ypade = (alpha[:,np.newaxis] * xdata / (1+beta[:,np.newaxis]*xdata)).sum(axis=0) * (xdata+1)
        ax.plot(xdata, ypade, label=f'pade({order})')
    ax.legend()
    ax.set_title('(1+x)log(1+x), fractional pade')
    fig.savefig('data/pade_xlog1p_fractional.png')
