import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import numpy as np
from numpy.linalg import norm
import scipy

from math_utils import solve_U, update_V, E, U_converged
from metrics import metric
from energy import Energy

from functools import reduce
import operator


class Anderson:

    def __init__(self, m, ndim, u0):

        self.m_ = m
        self.dim_ = ndim
        # self.current_u_ = np.zeros((1, ndim))
        self.current_F_ = np.zeros((ndim,))
        self.prev_dG_ = np.zeros((ndim, m))
        self.prev_dF_ = np.zeros((ndim, m))
        self.M_ = np.zeros((m, m))
        self.theta_ = np.zeros((m, ))
        self.G = np.zeros((1, ndim))
        self.dF_scale_ = np.zeros((m, ))
        self.current_u_ = u0
        self.iter_ = 0
        self.col_idx_ = 0

    def set_G(self, g):
        self.G = g

    def replace(self, u):

        self.current_u_ = u

    def compute(self):

        self.current_F_ = self.G - self.current_u_
        ndim = self.dim_

        if self.iter_ == 0:
            self.prev_dF_[:, 0] = -self.current_F_
            self.prev_dG_[:, 0] = -self.G
            current_u_ = self.G
        else:
            self.prev_dF_[:, self.col_idx_] += self.current_F_.reshape((ndim,))
            self.prev_dG_[:, self.col_idx_] += self.G.reshape((ndim,))

            eps = 1e-14
            scale = np.max([eps, norm(self.prev_dF_[:, self.col_idx_])])
            self.dF_scale_[self.col_idx_] = scale
            self.prev_dF_[:, self.col_idx_] /= scale

            m_k = np.min([self.m_, self.iter_])

            if m_k == 1:
                self.theta_[0] = 0
                dF_norm = norm(self.prev_dF_[:, self.col_idx_])
                dF_sqrnorm = dF_norm ** 2
                self.M_[0:0] = dF_sqrnorm

                if dF_norm > eps:
                    self.theta_[0] = (self.prev_dF_[:, self.col_idx_] * self.current_F_).sum() / dF_sqrnorm
            else:
                new_inner_prod = (self.prev_dF_[:, self.col_idx_].T @ self.prev_dF_[:, 0: m_k]).T
                self.M_[self.col_idx_, 0:m_k] = new_inner_prod.T
                self.M_[0:m_k, self.col_idx_] = new_inner_prod

                self.theta_[0:m_k] = scipy.linalg.solve(self.M_[0:m_k, 0:m_k], self.prev_dF_[:, 0:m_k].T @ self.current_F_.reshape((ndim, 1))).reshape(m_k)
            current_u_ = self.G - self.prev_dG_[:, 0:m_k] @ (self.theta_[0:m_k] / self.dF_scale_[0:m_k]).T

            self.col_idx_ = (self.col_idx_ + 1) % self.m_
            self.prev_dF_[:, self.col_idx_] = -self.current_F_
            self.prev_dG_[:, self.col_idx_] = -self.G

        self.current_u_ = current_u_
        self.iter_ += 1

        return current_u_


def anderson_iteration(X, U, V, labels, p, logger):

    t = 0
    mmax = p.mmax or 3

    gamma = p.gamma
    epsilon = p.epsilon
    max_iteration = p.max_iterations or 300

    V_old = update_V(V, U, X, epsilon)
    U_old = U

    old_E = np.Infinity
    VAUt = None

    aa_ndim = reduce(operator.mul, V.shape)
    aa_shape = (1, aa_ndim)
    accelerator = None

    energy = Energy()

    while True:
        U_new = solve_U(X, V_old, gamma, epsilon)

        # delta_U, is_converged = U_converged(U_new, U_old)

        new_E = E(U_new, V_old, X, gamma, epsilon)

        if new_E >= old_E:
            V_old = VAUt
            U_new = solve_U(X, V_old, gamma, epsilon)
            new_E = E(U_new, V_old, X, gamma, epsilon)

            accelerator.replace(V_old.reshape(aa_shape))

        if energy.converged(new_E):
            return U_new, V_old, t, metric(U_new, labels)

        if t > max_iteration:
            return U_new, V_old, t, metric(U_new, labels)

        energy.add(new_E)
        logger.log_middle(new_E, metric(U_new, labels))

        VAUt = update_V(V_old, U_new, X, epsilon)

        if t == 0:
            accelerator = Anderson(mmax, aa_ndim, VAUt.reshape(aa_shape))
            V_new = VAUt
        else:
            accelerator.set_G(VAUt.reshape(aa_shape))
            V_new = accelerator.compute().reshape(V.shape)

        delta_V, _ = U_converged(V_new, V_old)
        V_old = V_new
        # U_old = U_new
        # old_E = new_E
        t += 1
