import itertools
import math
import os.path
import pickle
import random
import struct
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps

import numpy as np
import portalocker
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as FF
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skorch.utils import to_numpy
from torch.profiler import record_function
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import *
# %% Define Tool Functions

@contextmanager
def conditional_no_grad(no_grad=True):
    if no_grad:
        with torch.no_grad():
            yield
    else:
        yield

def batch_process(func, X, batch_size=16, no_grad=True):
    with conditional_no_grad(no_grad=no_grad):
        outs = []
        DL = TensorLoader(X, X, batch_size, shuffle=False)
        for x, _ in DL:
            outs.append(func(x))
        return torch.cat(outs)


def iselect(source, index, dim=0):
    # 1) If the index is a list, convert it to a tensor
    if isinstance(index, list):
        index = torch.tensor(index, dtype=torch.long, device=source.device)

    # Ensure that the index is on the same device as the source tensor
    index = index.to(source.device)

    # 2) Perform the index_select operation
    return source.index_select(dim, index)


class TensorLoader:
    def __init__(self, X, Y, batch_size, shuffle=False, origin_indices=None, dropend=False, attr=None):
        self.X = X
        if Y is None:
            Y = torch.zeros(len(X), device=X.device)
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_index = 0
        self.dropend = dropend
        if origin_indices is None:
            origin_indices = torch.arange(len(X))

        self.length = len(origin_indices)
        self.origin_indices = origin_indices
        self.indices = origin_indices
        self.attr = attr

    def clone(self):
        shuffle = self.shuffle
        self.shuffle = False
        TL = TensorLoader(self.get_X(), self.get_Y(), self.batch_size, shuffle)
        self.shuffle = shuffle
        return TL

    def get_random_remove(self, num_remove, remove_seed, return_removed=False):
        XX = self.get_X()
        YY = self.get_Y()
        generator = torch.Generator().manual_seed(remove_seed)

        classes = torch.unique(YY)
        remaining_indices = []
        removed_indices = []

        for class_label in classes:
            # Get indices of samples with this class
            class_indices = torch.where(YY == class_label)[0]
            # Randomly select indices to remove
            perm = torch.randperm(len(class_indices), generator=generator)
            remove_indices = class_indices[perm[:num_remove]]

            # Keep track of removed indices
            removed_indices.append(remove_indices)
            # Keep the remaining indices
            remaining_indices.append(class_indices[perm[num_remove:]])

        # Concatenate and sort to maintain original order (minus removed samples)
        remaining_indices = torch.cat(remaining_indices).sort().values
        removed_indices = torch.cat(removed_indices).sort().values

        # Filter the original tensors
        XX_remaining = XX[remaining_indices]
        YY_remaining = YY[remaining_indices]

        XX_removed = XX[removed_indices]
        YY_removed = YY[removed_indices]

        if return_removed:
            return (
                TensorLoader(XX_remaining, YY_remaining, self.batch_size, self.shuffle),
                TensorLoader(XX_removed, YY_removed, self.batch_size, self.shuffle)
            )
        else:
            return TensorLoader(XX_remaining, YY_remaining, self.batch_size, self.shuffle)

    def get_X(self):
        # return self.X.index_select(0, self.origin_indices)
        return iselect(self.X, self.origin_indices)
        # return self.X[self.origin_indices]

    def get_Y(self):
        return iselect(self.Y, self.origin_indices)
        # return self.Y[self.origin_indices]

    def to(self, device):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        if isinstance(self.attr, torch.Tensor):
            self.attr.to(device)

    def cuda(self, device=None):
        self.X = self.X.cuda(device)
        self.Y = self.Y.cuda(device)
        if isinstance(self.attr, torch.Tensor):
            self.attr.cuda(device)
        return self

    def cpu(self):
        self.X = self.X.cpu()
        self.Y = self.Y.cpu()
        if isinstance(self.attr, torch.Tensor):
            self.attr.cpu()
        return self

    def augment(self, augfunc, augsize, equal=True, C=None, as_XY=False):
        if C is None:
            C = self.Y.amax().item() + 1
        assert augsize % C == 0
        augsize_per = augsize // C
        xs = {y: None for y in range(C)}
        shuffle = self.shuffle
        self.shuffle = True
        while sum([0 if x is None else len(x) for x in xs.values()]) < augsize:
            x, y = next(iter(self))
            for i in range(C):
                ax = augfunc(x)
                if xs[i] is None:
                    xs[i] = ax[y == i]
                else:
                    xs[i] = torch.cat([xs[i], ax[y == i]], dim=0)
                if equal:
                    xs[i] = xs[i][:augsize_per]

        self.shuffle = shuffle
        Y = torch.tensor(sum([[y] * len(xs[y]) for y in range(C)], []), device=self.Y.device)
        X = torch.cat([xs[y] for y in range(C)], dim=0)
        if as_XY:
            return X, Y
        else:
            return TensorLoader(
                torch.cat([self.get_X(), X], dim=0),
                torch.cat([self.get_Y(), Y], dim=0),
                self.batch_size,
                self.shuffle,
                dropend=self.dropend,
            )

    def split(self, left_ratio=0.8, seed=0, left_batchsize=None, left_shuffle=None, right_batchsize=None,
              right_shuffle=None):
        cur_state = torch.random.get_rng_state()
        torch.random.manual_seed(seed)
        left_length = int(self.length * left_ratio)
        randind = self.origin_indices[torch.randperm(self.length)]
        left_ind = randind[:left_length]
        right_ind = randind[left_length:]
        left_TL = TensorLoader(self.X, self.Y,
                               left_batchsize or self.batch_size,
                               left_shuffle or self.shuffle,
                               left_ind)
        right_TL = TensorLoader(self.X, self.Y,
                                right_batchsize or self.batch_size,
                                right_shuffle or self.shuffle,
                                right_ind)
        torch.random.set_rng_state(cur_state)
        return left_TL, right_TL

    def CVsplit(self, total_cv, seed=None):
        kfold = StratifiedKFold(total_cv, shuffle=True, random_state=seed)
        yy = to_numpy(self.Y)
        TDLs = []
        VDLs = []
        for tind, vind in kfold.split(self.origin_indices, yy):
            TDLs.append(TensorLoader(self.X, self.Y, self.batch_size, True, torch.tensor(tind).long()))
            VDLs.append(TensorLoader(self.X, self.Y, self.batch_size, False, torch.tensor(vind).long()))
        return TDLs, VDLs

    def __iter__(self):
        if self.shuffle:
            self.indices = self.origin_indices[torch.randperm(self.length)]
        else:
            self.indices = self.origin_indices
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.length:
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        if self.dropend and len(batch_indices) < self.batch_size:
            raise StopIteration
        self.current_index += self.batch_size
        if self.attr == "index":
            return (iselect(self.X, batch_indices), batch_indices), iselect(self.Y, batch_indices)
        elif self.attr is not None:
            return (iselect(self.X, batch_indices), iselect(self.attr, batch_indices)), iselect(self.Y, batch_indices)
        else:
            return iselect(self.X, batch_indices), iselect(self.Y, batch_indices)
        # return self.X[batch_indices], self.Y[batch_indices]

    def get_XY(self):
        last_batch = self.batch_size
        self.batch_size = len(self.X)
        x, y = next(iter(self))
        self.batch_size = last_batch
        return x, y


def safelog(x):
    return torch.log(torch.clamp(x, 1e-6, 1e6))


def manual_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # noqa
    torch.backends.cudnn.benchmark = False  # noqa
# %% Define CSP-LDA Base Functions

def calc_fr(F1, F2, axis=-1, oneway=False):
    m1 = F1.mean(axis=axis, keepdims=True)
    m2 = F2.mean(axis=axis, keepdims=True)
    v1 = F1.var(axis=axis, keepdims=True)
    v2 = F2.var(axis=axis, keepdims=True)
    vv = (v1 + v2).clamp_min(1e-6)
    if oneway:
        return (torch.sign(m1 - m2) * (m1 - m2) ** 2 / vv).mean(axis=axis)
    else:
        return ((m1 - m2) ** 2 / vv).mean(axis=axis)


def GI(ori_M, K, tol=1e-6, maxiter=50):
    eigs = []
    eigvs = []
    for k in range(K):
        M = ori_M
        M = M / M.amax(dim=[-1, -2], keepdims=True)
        last_x = M.sum(dim=-1, keepdims=True)
        for n_iter in range(maxiter):
            M = M @ M
            M = M / M.amax(dim=[-1, -2], keepdims=True)
            x = M.sum(dim=-1, keepdims=True)
            if (x - last_x).abs().max() < tol:
                break
            last_x = x
            # res.append((x.T@M@x).item())
        x = last_x
        xT = x.transpose(-1, -2)
        xTx = xT @ x
        l1 = (xT @ ori_M @ x) / xTx
        # l1 = (last_x.T @ ori_M @ last_x) / (last_x.T @ last_x)
        # l1 = (last_x.T @ A @ last_x) / (last_x.T @ B @ last_x)
        # ori_M = ori_M - l1*(last_x@last_x.T)/(last_x.T@last_x)
        ori_M = ori_M - l1 * (x @ xT) / (xTx)
        eigs.append(l1.squeeze(-1))
        eigvs.append(x)
    # eigs: ... x K
    # eigvs: ... x  n_ch X K
    eigs, eigvs = torch.cat(eigs, dim=-1), torch.cat(eigvs, dim=-1)
    # print(eigvs.shape)
    return eigs, eigvs.transpose(-1, -2)


def solve_GRQ(A, B, method="pi", niter=10, k=1):
    # Solve generalized rayleigh quotient
    if method == "eig":
        assert A.ndim == 3
        n_f, n_ch, _ = A.shape
        n_size = n_f * n_ch
        M = torch.linalg.solve(B, A)
        eig_values, u_mat = torch.linalg.eig(M)
        sort_indices = torch.argsort(eig_values.abs(), descending=True, dim=1)
        indices_fix = torch.arange(n_f, device=A.device) * n_ch
        fixed_sort_indices = (sort_indices + indices_fix[:, None]).reshape(-1)
        u_mat = torch.transpose(u_mat, 1, 2).real
        u_mat = u_mat.reshape(n_size, n_ch)[fixed_sort_indices].reshape(n_f, n_ch, n_ch)
        return u_mat[:, :k]
    elif method == "pi":
        M = torch.linalg.solve(B, A)
        return GI(M, k, 1e-6, niter)[1]


def calc_CSP_filter(m0, m1, k=1, niter=50, eps=0.):
    if eps > 0:
        eps = torch.eye(m0.shape[-1], device=m0.device) * eps
    w0 = solve_GRQ(m0, m1 + eps, niter=niter, k=k)
    w1 = solve_GRQ(m1, m0 + eps, niter=niter, k=k)
    return torch.cat([w0, w1], dim=-2)


def apply_CSP_filter(ws, X, do_log=True, select_k=None):
    if select_k is not None:
        n_s = ws.shape[-2]
        n_half = n_s // 2
        ws = torch.cat([ws[..., :select_k, :], ws[..., n_half:n_half + select_k, :]], dim=-2)
    f = ws[..., None, :] @ X[..., None, :, :] @ ws[..., None]
    f = f[..., 0, 0]
    if do_log:
        f = safelog(f)
    return f


def calc_lda_filter_pre(X, Y):
    """
    Pre-calculation for multi-view LDA.

    Computes necessary statistics by removing the batch dimension, and reshapes them to the original view dimensions.
    Returns:
        stats (dict): Dictionary containing:
            - m0 (torch.Tensor): Mean for class 0, reshaped to (*views, n_f)
            - m1 (torch.Tensor): Mean for class 1, reshaped to (*views, n_f)
            - S_w (torch.Tensor): Within-class scatter matrix per view, reshaped to (*views, n_f, n_f)
    """
    n_b = X.shape[0]
    view_shape = X.shape[1:-1]  # extra dimensions (views)
    n_f = X.shape[-1]

    # Compute total number of view elements
    N = 1
    for d in view_shape:
        N *= d

    # Reshape X to (n_b, N, n_f) so that each view is processed independently.
    X_flat = X.reshape(n_b, N, n_f)

    # Boolean masks for the two classes (assumed to be 0 and 1)
    mask0 = (Y == 0)
    mask1 = (Y == 1)

    # Separate samples by class
    X0 = X_flat[mask0]  # shape: (n0, N, n_f)
    X1 = X_flat[mask1]  # shape: (n1, N, n_f)

    # Compute class means for each view (shape: (N, n_f))
    m0 = X0.mean(dim=0)
    m1 = X1.mean(dim=0)

    # Compute within-class scatter matrices for each view without using unsqueeze(0)
    # d0: (n0, N, n_f) and we compute outer products for each view element:
    d0 = X0 - m0  # m0 broadcasts to (n0, N, n_f)
    S0 = (d0[..., None] * d0[..., None].transpose(-1, -2)).sum(dim=0)  # shape: (N, n_f, n_f)

    d1 = X1 - m1
    S1 = (d1[..., None] * d1[..., None].transpose(-1, -2)).sum(dim=0)  # shape: (N, n_f, n_f)

    # Total within-class scatter for each view
    S_w = S0 + S1  # shape: (N, n_f, n_f)

    # Reshape m0, m1 and S_w back to the original view dimensions.
    m0 = m0.reshape(*view_shape, n_f)
    m1 = m1.reshape(*view_shape, n_f)
    S_w = S_w.reshape(*view_shape, n_f, n_f)

    stats = {
        "m0": m0,
        "m1": m1,
        "S_w": S_w,
    }
    return stats


def calc_lda_filter_after(stats):
    """
    Post-calculation for multi-view LDA.

    Computes the LDA filter w and bias b from pre-computed statistics that are already reshaped
    to the original view dimensions.

    Args:
        stats (dict): Dictionary containing:
            - m0 (torch.Tensor): Mean for class 0, shape (*views, n_f)
            - m1 (torch.Tensor): Mean for class 1, shape (*views, n_f)
            - S_w (torch.Tensor): Within-class scatter matrices, shape (*views, n_f, n_f)

    Returns:
        tuple: (w, b)
            - w (torch.Tensor): LDA filter of shape (*views, n_f)
            - b (torch.Tensor): Bias term of shape (*views,)
    """
    m0 = stats["m0"]  # shape: (*views, n_f)
    m1 = stats["m1"]  # shape: (*views, n_f)
    S_w = stats["S_w"]  # shape: (*views, n_f, n_f)

    # Compute the difference between class means for each view.
    diff = (m1 - m0)[..., None]  # shape: (*views, n_f, 1)

    # Add a small regularization term for numerical stability.
    n_f = m0.shape[-1]
    reg = 1e-4 * torch.eye(n_f, device=S_w.device)
    # Solve for w using the LDA formulation: S_w * w = diff, then remove the extra dimension.
    w = torch.linalg.solve(S_w + reg, diff)[..., 0]  # shape: (*views, n_f)

    # Compute the projected means using the precomputed m0 and m1.
    proj0 = (m0 * w).sum(dim=-1)  # shape: (*views,)
    proj1 = (m1 * w).sum(dim=-1)  # shape: (*views,)
    pdiff = (proj1 - proj0).clamp(min=1e-5)

    # Compute the scaling factor per view so that the difference becomes 2.
    a = 2.0 / pdiff  # shape: (*views,)

    # Incorporate the scaling into w.
    w = a[..., None] * w  # shape: (*views, n_f)

    # Compute the bias term so that the normalized projected mean for class 0 becomes -1.
    b = -1.0 - a * proj0  # shape: (*views,)

    return w, b


def calc_lda_filter(X, Y):
    """
    Full calculation of multi-view LDA filter and bias.

    First computes the necessary statistics and then calculates w and b.
    """
    stats = calc_lda_filter_pre(X, Y)
    return calc_lda_filter_after(stats)


def calc_lda_filter_parallal(Xs, Ys):
    m0 = []
    m1 = []
    S_w = []
    for X, Y in zip(Xs, Ys):
        stats = calc_lda_filter_pre(X, Y)
        m0.append(stats['m0'])
        m1.append(stats['m1'])
        S_w.append(stats['S_w'])
    m0 = torch.stack(m0, dim=0)
    m1 = torch.stack(m1, dim=0)
    S_w = torch.stack(S_w, dim=0)
    return calc_lda_filter_after({
        "m0": m0,
        "m1": m1,
        "S_w": S_w,
    })


def apply_lda_filter(X, w, b):
    return (X * w).sum(dim=-1) + b

# %% Define Neural Networks and NN-Trainer
class BlockSpatialFilter(nn.Module):
    def __init__(self, n_v, n_f, n_t, n_s, n_c, maxnorm=1., spfilter_norm=False):
        super().__init__()
        w = torch.randn(n_v, n_f, n_t, n_s, n_c) / 10
        self.w = nn.Parameter(w)
        self.spfilter_norm = spfilter_norm
        self.maxnorm = maxnorm

    def renorm(self):
        maxnorm = self.maxnorm
        if maxnorm > 999999:
            return
        with torch.no_grad():
            w = self.w.data
            n_v, n_f, n_t, n_s, n_c = w.shape
            w = w.reshape(n_v * n_f * n_t * n_s, n_c)
            w = torch.renorm(w, p=2, dim=0, maxnorm=maxnorm)
            w = w.reshape(n_v, n_f, n_t, n_s, n_c)
            self.w.data = w

    def forward(self, x):
        if x.ndim == 5:
            x = x[:, None]
        self.renorm()
        n_b, n_v, n_f, n_t, n_c, _ = x.shape
        x = x.reshape(n_b, n_v, n_f, n_t, 1, n_c, n_c)
        w = self.w
        n_v, n_f, n_t, n_s, n_c = w.shape
        if self.spfilter_norm:
            w = w / w.norm(dim=-1, keepdim=True, p=2)
        x = (w[..., None, :] @ x @ w[..., :, None])[..., 0, 0]
        n_b, n_v, n_f, n_t, n_s = x.shape
        return x


class BatchNormCov(nn.Module):
    def __init__(self, n_v, n_f, n_t, n_s, momentum=0.1):
        super().__init__()
        # n_t == 1
        self.la = nn.Parameter(
            torch.ones(n_v, n_f, n_t, n_s),
            requires_grad=False
        )
        self.momentum = momentum

    def forward(self, x):
        n_b, n_v, n_f, n_t, n_s = x.shape
        if self.training:
            # sx = x.mean(dim=-2, keepdim=True)
            sx = x.mean(dim=0)
            cur_la = sx
            with torch.no_grad():
                self.la.data = self.la.data * (1 - self.momentum) + cur_la * self.momentum
        else:
            cur_la = self.la
        x = x / cur_la
        return x


class BlockVoteFCCLF(nn.Module):
    def __init__(self, n_v, n_f, n_t, n_s, n_class, bias=True, maxnorm=1., reg_share=False, dropout_rate=0.2):
        super().__init__()
        w = torch.randn(n_v, n_class, n_f, n_t, n_s) / 10
        b = torch.randn(n_v, n_class, n_f, n_t) / 10
        self.bias = bias
        self.w = nn.Parameter(w, requires_grad=True)
        self.b = nn.Parameter(b, requires_grad=True)
        self.reg_share = reg_share
        self.maxnorm = maxnorm
        self.drop = nn.Dropout(p=dropout_rate)

    def renorm(self):
        maxnorm = self.maxnorm
        if maxnorm > 999999:
            return
        with torch.no_grad():
            w = self.w.data
            n_v, n_class, n_f, n_t, n_s = w.shape
            if self.reg_share:
                w = w.reshape(n_v * n_class * n_f, n_t * n_s)
            else:
                w = w.reshape(n_v * n_class * n_f * n_t, n_s)
            w = torch.renorm(w, p=2, dim=0, maxnorm=maxnorm)
            w = w.reshape(n_v, n_class, n_f, n_t, n_s)
            self.w.data = w

    def forward(self, x):
        self.renorm()
        x = self.drop(x)
        n_b, n_v, n_f, n_t, n_s = x.shape
        w = self.w
        bias = self.bias
        b = self.b
        n_v, n_class, n_f, n_t, n_s = w.shape
        x = x.reshape(n_b, n_v, n_f, n_t, 1, n_s, 1)
        w = w.permute(0, 2, 3, 1, 4)
        w = w.reshape(n_v, n_f, n_t, n_class, 1, n_s)
        x = (w @ x)[..., 0, 0]
        n_b, n_v, n_f, n_t, n_class = x.shape
        if bias:
            b = b.permute(0, 2, 3, 1)
            x = x + b
        return x


class MulticompNet(nn.Module):
    def __init__(self, n_g, n_cp, n_s, n_c, n_class, max_spnorm=1., max_clfnorm=1., dropout=0.5, ):
        """
        Input: n_b x n_g x n_cp x n_ch x n_ch

        statedict
            cov_sp_filter.w torch.Size([1, 5, 8, 32, 22])
            sp_bn.la torch.Size([1, 5, 8, 32])
            clf.w torch.Size([1, 2, 5, 1, 256])
            clf.b torch.Size([1, 2, 5, 1])
        """
        super().__init__()
        self.cov_sp_filter = BlockSpatialFilter(1, n_g, n_cp, n_s, n_c, max_spnorm)
        self.sp_bn = BatchNormCov(1, n_g, n_cp, n_s)
        self.clf = BlockVoteFCCLF(1, n_g, 1, n_cp * n_s, n_class,
                                  maxnorm=max_clfnorm, dropout_rate=dropout)

    def forward_logvar(self, x):
        n_b, n_g, n_cp, n_c, _ = x.shape
        x = self.cov_sp_filter(x[:, None])
        x = self.sp_bn(x)
        x = safelog(x)
        n_s = x.shape[-1]
        x = x.reshape(n_b, n_g, n_cp * n_s)
        return x

    def forward(self, x):
        x = self.forward_logvar(x)
        x = self.clf(x[:, None, :, None])[:, 0, :, 0]  # n_b x n_g x n_class
        return x

    @torch.no_grad()
    def para_zip(self, sd):
        # sd: state_dict
        _, n_g, n_cp, n_s, n_c = self.cov_sp_filter.w.shape
        _, n_class, _, _, _ = self.clf.w.shape
        p1 = sd['cov_sp_filter.w'].reshape(n_g, n_cp*n_s*n_c)
        p2 = sd['sp_bn.la'].reshape(n_g, n_cp*n_s)
        p3 = sd['clf.w'].reshape(n_class, n_g, n_cp*n_s).movedim(0, -1).reshape(n_g, n_cp*n_s*n_class)
        p4 = sd['clf.b'].reshape(n_class, n_g).movedim(0, -1)
        p = torch.cat((p1, p2, p3, p4), dim=-1)
        return p

    @torch.no_grad()
    def para_unzip(self, p):
        _, n_g, n_cp, n_s, n_c = self.cov_sp_filter.w.shape
        _, n_class, _, _, _ = self.clf.w.shape
        l1 = n_cp*n_s*n_c
        l2 = n_cp*n_s
        l3 = n_cp*n_s*n_class
        l4 = n_class
        s1, e1 = 0, l1
        s2, e2 = e1, e1+l2
        s3, e3 = e2, e2+l3
        s4, e4 = e3, e3+l4
        p1 = p[:, s1:e1].reshape(1, n_g, n_cp, n_s, n_c)
        p2 = p[:, s2:e2].reshape(1, n_g, n_cp, n_s)
        p3 = p[:, s3:e3].reshape(n_g, n_cp*n_s, n_class).movedim(-1, 0).reshape(1, n_class, n_g, 1, n_cp*n_s)
        p4 = p[:, s4:e4].movedim(-1, 0).reshape(1, n_class, n_g, 1)
        sd = {
            "cov_sp_filter.w": p1,
            "sp_bn.la": p2,
            "clf.w": p3,
            "clf.b": p4,
        }
        return sd

    def fit_with_valid_mv(self, TDL, VDL, max_epoch=1000, lr_start=2**-12, lr_end=2**-12, weight_decay=0.0, EDL=None, load_as_best=True, verbose=True):
        acc_aggfunc = lambda outs, ys: outs.argmax(dim=-1).eq(ys[:,None]).float().mean(dim=0)
        lossfunc = lambda out, y: calc_CE_loss(out, y, already_softmax=False)
        model = self
        train_forward_func = lambda x, y: (model(x), y)
        test_forward_func = train_forward_func
        para_zip = self.para_zip
        para_unzip = self.para_unzip
        rec = []
        optim = torch.optim.AdamW(model.parameters(), lr=lr_start, weight_decay=weight_decay)
        pbar = tqdm(total=max_epoch, disable=not verbose)
        r = math.exp(math.log(lr_end / lr_start) / max_epoch)
        cur_lr = lr_start

        tve_rec = {}  # train/valid/test outs/ys/acc/loss
        # train_outs = train_ys = train_acc = train_loss = None
        # test_outs = test_ys = test_acc = test_loss = None
        # valid_outs = valid_ys = valid_acc = valid_loss = None

        best_state = None
        best_losses = None
        n_views = self.cov_sp_filter.w.shape[1]  # n_g

        for epoch in range(max_epoch):
            optim.param_groups[0]['lr'] = cur_lr
            cur_lr *= r
            model.train()
            TDL.shuffle = True
            for x, y in TDL:
                out, y = train_forward_func(x, y)
                loss = lossfunc(out, y)
                loss = loss.mean(axis=0).sum()
                loss.backward()
                optim.step()
                optim.zero_grad()

            model.eval()
            TDL.shuffle = False
            _refresh = False
            with torch.no_grad():
                for typ, DL in zip(["valid","train","test"], [VDL, TDL, EDL]):
                    if DL is None:
                        # VDL is None: No Eval;
                        # VDL is not None: Eval when V Updating
                        break
                    outs, losses, ys = [], [], []
                    for x, y in DL:
                        out, y = test_forward_func(x, y)
                        loss = lossfunc(out, y)
                        outs.append(out)
                        ys.append(y)
                        losses.append(loss)
                    outs = torch.cat(outs, dim=0)
                    ys = torch.cat(ys, dim=0)
                    loss = torch.cat(losses, dim=0).mean(dim=0)

                    tve_rec[f"{typ}_acc"] = acc_aggfunc(outs, ys).detach().cpu()
                    tve_rec[f"{typ}_loss"] = loss.detach().cpu()
                    tve_rec[f"{typ}_ys"] = ys.detach().cpu()
                    tve_rec[f"{typ}_outs"] = outs.detach().cpu()

                    if typ == "valid":
                        # Update best loss
                        # loss: n_views
                        if best_state is None:
                            best_state = para_zip(model.state_dict())
                            best_losses = loss
                            _refresh = True
                            continue

                        mask = loss<best_losses
                        if mask.sum()>0:
                            para = para_zip(model.state_dict())
                            best_losses[mask] = loss[mask]
                            best_state[mask] = para[mask]
                            _refresh = True
                            continue

                        if not _refresh:
                            break

            if _refresh:
                # noinspection PyUnboundLocalVariable

                rec.append({
                    'epoch': epoch,
                    'train_loss': tve_rec["train_loss"],
                    'train_acc': tve_rec["train_acc"],
                    'test_loss': tve_rec["test_loss"],
                    'test_acc': tve_rec["test_acc"],
                    'valid_loss': tve_rec["valid_loss"],
                    'valid_acc': tve_rec["valid_acc"],
                })

                display_dict = {}
                if TDL is not None:
                    display_dict['T'] = fr"{tve_rec['train_acc'].amax().item()*100:.2f}"
                if VDL is not None:
                    display_dict['V'] = fr"{tve_rec['valid_acc'].amax().item() * 100:.2f}"
                if EDL is not None:
                    display_dict['E'] = fr"{tve_rec['test_acc'].amax().item() * 100:.2f}"
                pbar.set_postfix(display_dict, refresh=False)
            TDL.shuffle = True

            pbar.update()
        pbar.close()

        best_state = para_unzip(best_state)
        if load_as_best:
            model.load_state_dict(best_state)
        rec[-1].update({
            "train_outs": tve_rec["train_outs"],
            "train_ys": tve_rec["train_ys"],
            "test_outs": tve_rec["test_outs"],
            "test_ys": tve_rec["test_ys"],
            "valid_outs": tve_rec["valid_outs"],
            "valid_ys": tve_rec["valid_ys"],
            "best_state": best_state,
        })
        return rec

    def fit_with_retain_mv(self, ADL:TensorLoader, EDL:TensorLoader, total_cv=5, seed=0,
                     first_epoch=1000, second_epoch=500,
                     first_lr_start=2**-12, first_lr_end=2**-12,
                     second_lr_start=2**-12, second_lr_end=2**-12,
                     weight_decay=0.,
                     _rec=None,
                     verbose=True):
        TDLs, VDLs = ADL.CVsplit(total_cv, seed)
        VDL = TensorLoader(VDLs[0].get_X(), VDLs[0].get_Y(), ADL.batch_size, shuffle=False)
        TDL = TensorLoader(TDLs[0].get_X(), TDLs[0].get_Y(), ADL.batch_size, shuffle=True)
        if _rec is None:
            rec = self.fit_with_valid_mv(TDL, VDL, first_epoch, first_lr_start, first_lr_end, weight_decay, EDL, True, verbose=verbose)
        else:
            rec = _rec
        rec2 = self.fit_with_valid_mv(ADL, ADL, second_epoch, second_lr_start, second_lr_end, weight_decay, EDL, True, verbose=verbose)
        return rec2

def calc_CE_loss(f, y, already_softmax=False):
    n_classes = f.shape[-1]
    if already_softmax:
        f = torch.log(f)
    else:
        f = torch.log_softmax(f, dim=-1)
    mask = F.one_hot(y, n_classes)
    select_f = f * mask[:, *([None] * (f.ndim - mask.ndim)), :]
    # 3.8:
    # mask_shape = mask.shape
    # new_shape = mask_shape + (1,) * (f.ndim - mask.ndim)
    # mask_expanded = mask.reshape(new_shape)
    # select_f = f * mask_expanded

    select_f = select_f.sum(dim=-1)
    return -select_f


def fit_ce_loss_ex(model, TDL, max_epoch, lr_start=1e-3, lr_end=1e-5, warmup=10, EDL=None, eval_update=1,
                   acc_aggfunc=None, prehook=None, reduce_loss=True, lossfunc=None,
                   forwardfunc=None, custom_optim=None, custom_schedule=None, verbose=True, tqdm_kwargs=None):
    """
    if withattr:
        x, y in DL --> (x, attr), y in DL

    """
    if tqdm_kwargs is None:
        tqdm_kwargs = {}
    self = model
    if acc_aggfunc is None:
        acc_aggfunc = 2
    if isinstance(acc_aggfunc, int):
        ss = [slice(None)] + [None] * acc_aggfunc
        acc_aggfunc = lambda outs, ys: outs.argmax(dim=-1).eq(ys[ss]).float().mean(dim=0)
    if prehook is None:
        prehook = lambda env: None

    if lossfunc is None:
        lossfunc = lambda rawout, y: (rawout, calc_CE_loss(rawout, y, already_softmax=False))
    if forwardfunc is None:
        forwardfunc = lambda x, y: model(x)
    if custom_optim is None:
        optim = torch.optim.Adam(self.parameters(), lr=lr_start)
    else:
        optim = custom_optim
    if custom_schedule is None:
        lr_list = torch.logspace(math.log10(lr_start), math.log10(lr_end), max_epoch)
        if warmup > 0:
            lr_list[:warmup] = torch.linspace(0., lr_list[warmup], warmup)

        def _default_schedule(epo, m_epo):
            optim.param_groups[0]["lr"] = lr_list[epo].item()

        lr_schedule = _default_schedule
    else:
        lr_schedule = custom_schedule
    rec = []
    pbar = tqdm(total=max_epoch, disable=not verbose, **tqdm_kwargs)
    for epoch in range(max_epoch):
        env = {"epoch": epoch, "max_epoch": max_epoch, "model": model}
        prehook(env)
        lr_schedule(epoch, max_epoch)
        self.train()
        TDL.shuffle = True
        for x, y in TDL:
            out = forwardfunc(x, y)
            out, loss = lossfunc(out, y)
            loss = loss.mean(axis=0).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()

        if EDL is not None and (epoch + 1) % eval_update == 0:
            self.eval()
            TDL.shuffle = False
            with torch.no_grad():
                for stage in [0, 1]:
                    outs, losses, ys = [], [], []
                    for x, y in TDL if stage == 0 else EDL:
                        out = forwardfunc(x, y)
                        out, loss = lossfunc(out, y)
                        outs.append(out)
                        ys.append(y)
                        losses.append(loss)
                    outs = torch.cat(outs, dim=0)
                    ys = torch.cat(ys, dim=0)
                    losses = torch.cat(losses, dim=0)
                    if reduce_loss:
                        losses = losses.mean(dim=0)
                    if stage == 0:
                        train_acc = acc_aggfunc(outs, ys).detach().cpu()
                        train_loss = losses.detach().cpu()
                        train_ys = ys.detach().cpu()
                        train_outs = outs.detach().cpu()
                    else:
                        test_acc = acc_aggfunc(outs, ys).detach().cpu()
                        test_loss = losses.detach().cpu()
                        test_ys = ys.detach().cpu()
                        test_outs = outs.detach().cpu()

            # noinspection PyUnboundLocalVariable
            rec.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            })

            pbar.set_postfix({
                "T": fr"{train_acc.amax() * 100:.2f}%",
                "E": fr"{test_acc.amax() * 100:.2f}%"
            }, refresh=False)

        pbar.update()
    pbar.close()
    if EDL is not None:
        rec[-1].update({
            "train_outs": train_outs,
            "train_ys": train_ys,
            "test_outs": test_outs,
            "test_ys": test_ys,
        })
        return rec
    else:
        return None

# %% Define FIR and IIR Filters

class SimpleFreqSincBanks(nn.Module):
    def __init__(self, Fs, bands, n):
        super().__init__()
        self.n_out = len(bands)
        self.Fs = Fs
        lows = torch.tensor([x[0] for x in bands])
        highs = torch.tensor([x[1] for x in bands])
        lows = lows / Fs
        highs = highs / Fs
        self.lows = nn.Parameter(lows, requires_grad=False)
        self.highs = nn.Parameter(highs, requires_grad=False)
        self._win = torch.hamming_window(n)
        self.n = n

    @staticmethod
    def sinc_bandpass(low, high, fs, n_tap, win):
        # low/high: (n_filters, )
        # win: (n_tap, )
        # return: n_filters x n_tap
        nyq = 0.5 * fs
        low = low[:, None] / nyq
        high = high[:, None] / nyq
        alpha = 0.5 * (n_tap - 1)
        m = torch.arange(n_tap, device=low.device) - alpha
        m = m[None, :]
        h = high * torch.sinc(high * m) - low * torch.sinc(low * m)
        h = h * win[None, :]
        return h

    def get_filter(self):
        if self._win.device != self.lows.device:
            self._win = self._win.to(self.lows.device)
        Fs = self.Fs
        lows = self.lows * Fs
        highs = self.highs * Fs
        win = self._win
        n = self.n
        return self.sinc_bandpass(lows, highs, Fs, n, win)[:, None, None, :]

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, None]
        filt = self.get_filter()
        n = self.n
        x = F.conv2d(x, filt, padding=(0, n // 2))
        return x


def FIR_filter(x, Fs, bands, n_FIR):
    n_b, n_c, n_p = x.shape
    fbands = SimpleFreqSincBanks(Fs, bands, n_FIR).to(x.device)
    x = fbands(x)  # n_b x n_f x n_c x n_p
    return x


def IIR_filter(x, Fs, bands, n):
    sos_list = []
    for lowcut, highcut in bands:
        sos = signal.butter(n, [lowcut / (Fs / 2), highcut / (Fs / 2)], btype='band', output='sos')
        sos_list.append(torch.tensor(sos))
    sos_list = torch.stack(sos_list)

    a_mat = []
    b_mat = []

    for sos in sos_list:
        a_list = []
        b_list = []
        for section in sos:
            b_list.append(section[:3])
            a_list.append(section[3:])
        a_list = torch.stack(a_list)
        b_list = torch.stack(b_list)
        a_mat.append(a_list)
        b_mat.append(b_list)
    a_mat = torch.stack(a_mat).transpose(0, 1).float().to(x.device)
    b_mat = torch.stack(b_mat).transpose(0, 1).float().to(x.device)

    x = torch.stack([x] * len(bands), dim=-2)
    std_orig = x.std(dim=-1, keepdims=True).clamp(min=1e-8)
    x = x / std_orig

    for a_sec, b_sec in zip(a_mat, b_mat):
        x = FF.lfilter(x, a_sec, b_sec, False, True)

    x = x * std_orig
    x = x.movedim(-2, 1)
    return x
# %% Define ptfca
tf_type = Tuple[float, float, float, float]
tfg_type = Tuple[int, int, int, int]
tflist_type = List[tf_type]
tfglist_type = List[tfg_type]
cvsetting_type = Tuple[List[torch.Tensor], List[torch.Tensor]]

class LFUDict:
    def __init__(self, max_length):
        self.max_length = max_length
        self.data = {}
        self.freq = {}
        self.min_pair = (None, 0)

    def get(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return None

    def _update_min_pair(self):
        key = min(self.data, key=self.freq.get, default=None)
        freq = self.freq[key]
        return key, freq

    def update(self, key, value):
        if key in self.data:
            self.freq[key] += 1
        else:
            if key not in self.freq:
                self.freq[key] = 0
            self.freq[key] += 1
            cur_freq = self.freq[key]
            if len(self.data) < self.max_length:
                self.data[key] = value
            else:
                # One should be removed, or this should not be added
                if self.min_pair[0] is None:
                    self.min_pair = self._update_min_pair()
                if cur_freq <= self.min_pair[1]:
                    # This should not be added
                    pass
                else:
                    del self.data[self.min_pair[0]]
                    self.data[key] = value
                    self.min_pair = self._update_min_pair()
class PTFCA:
    def __init__(self, fs, n_c, n_p,
                 t_min_s=0.,
                 t_max_s=4.,
                 f_min_hz=4.,
                 f_max_hz=40.,
                 min_windowlength_s=0.2,
                 max_windowlength_s=4.,
                 min_bandwidth_hz=4.,
                 max_bandwidth_hz=36.,
                 t_stride_s=0.1,
                 f_stride_hz=1.,
                 f_mode="lasyexact",
                 f_use="FIR",  # FIR / IIR
                 n_FIR=91,
                 n_IIR=5,
                 checked_ks=(1, 2, 3, 4),
                 checked_tkregs=(0.,),
                 time_max_step=6,
                 freq_max_step=3,
                 time_step_multiple=(4, 2, 1),
                 freq_step_multiple=(4, 2, 1),
                 cd_maxiter=10,
                 overlap_gamma=1.,
                 overlap_mode="max",
                 random_init_timewindow_range_s=(0.5, 1.5),
                 random_init_timewindow_per_band=2,
                 als_CV=5,
                 criterion="AFR",
                 bandnorm=True,
                 ):
        self.fs = fs
        self.n_c = n_c
        self.n_p = n_p
        self.t_min_s = t_min_s
        if n_p>0 and t_max_s > n_p / fs:
            t_max_s = n_p / fs
        self.t_max_s = t_max_s
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.min_windowlength_s = min_windowlength_s
        self.max_windowlength_s = max_windowlength_s
        self.min_bandwidth_hz = min_bandwidth_hz
        self.max_bandwidth_hz = max_bandwidth_hz
        self.time_max_step = time_max_step
        self.freq_max_step = freq_max_step
        self.t_stride_s = t_stride_s
        self.p_stride = int(fs * t_stride_s)
        self.f_stride_hz = f_stride_hz
        self.als_CV = als_CV
        self.criterion = criterion

        assert n_p % self.p_stride == 0

        self.time_step_multiple = time_step_multiple
        self.freq_step_multiple = freq_step_multiple
        if random_init_timewindow_range_s[1]>max_windowlength_s:
            random_init_timewindow_range_s = (random_init_timewindow_range_s[0], max_windowlength_s)
        self.random_init_timewindow_range_s = random_init_timewindow_range_s
        self.random_init_timewindow_per_band = random_init_timewindow_per_band

        self.cd_maxiter = cd_maxiter

        self.f_mode = f_mode

        self.n_FIR = n_FIR
        self.n_IIR = n_IIR
        self.f_use = f_use
        self.overlap_gamma = overlap_gamma
        self.overlap_mode = overlap_mode
        self.checked_ks = checked_ks
        self.checked_tkregs = checked_tkregs
        self.caches = {}
        self.runtime = {}
        self._last_count_filter = 0
        self.count_afr = []
        self.count_filter = []
        self.runtime_shots = []

        self._use_cache_in_training = False
        self._max_exact_cache = 128
        self._trace_normalize = False
        self._band_normalize = bandnorm

        self.manual_cv_settings = None

        self._info = {}  # For checkpoints
        self.clear_cache()

    @staticmethod
    def calc_runtime(name=""):
        def decorator(func):
            def wrapper(self, *args, **kwargs):  # Ensure `self` is passed correctly
                if name not in self.runtime:
                    self.runtime[name] = 0
                start_time = time.time()
                result = func(self, *args, **kwargs)
                end_time = time.time()
                self.runtime[name] += end_time - start_time
                return result

            return wrapper

        return decorator

    @staticmethod
    def _create_cv_settings(n_b, y, total_cv, cv_seed=0):
        # Return: List[TDL], List[VDL]
        kfold = StratifiedKFold(total_cv, shuffle=True, random_state=cv_seed)
        y = to_numpy(y)
        tinds, vinds = [], []
        for tind, vind in kfold.split(np.arange(n_b), y):
            tinds.append(tind)
            vinds.append(vind)
        return tinds, vinds

    @record_function("filter")
    @calc_runtime("filter")
    def _filter(self, X, bands):
        self._last_count_filter += len(bands)
        fs = self.fs
        f_use = self.f_use
        if f_use == "FIR":
            return FIR_filter(X, fs, bands, self.n_FIR)
        elif f_use == "IIR":
            return IIR_filter(X, fs, bands, self.n_IIR)
        else:
            raise Exception("Unknown filter type")

    def get_tfcov_mean_lazyexact(self, tfgs, X, use_cache=True, return_cov=True):
        if use_cache:
            if "le_cache" not in self.caches:
                self.caches["le_cache"] = LFUDict(max_length=self._max_exact_cache)  # (2G for 200*22*1000)
            cache = self.caches["le_cache"]
        else:
            cache = None

        cache: Optional[LFUDict]

        cur_cache = {}
        for tg0, tg1, fg0, fg1 in tfgs:
            if (fg0, fg1) not in cur_cache:
                if cache is not None:
                    cur_cache[(fg0, fg1)] = cache.get((fg0, fg1))
                else:
                    cur_cache[(fg0, fg1)] = None

        gbands = [k for k, v in cur_cache.items() if v is None]
        if len(gbands) > 0:
            subbands = [(max(self.grid2freq(g0), 0.5), self.grid2freq(g1)) for g0, g1 in gbands]
            MAX_PROCESSED_SUBBANDS = 16
            _subbands = subbands.copy()
            _gbands = gbands.copy()
            while len(_subbands)>0:
                _X = self._filter(X, _subbands[:MAX_PROCESSED_SUBBANDS])
                for ind, (fg0, fg1) in enumerate(_gbands[:MAX_PROCESSED_SUBBANDS]):
                    bX = _X[:, ind]
                    if self._band_normalize:
                        if (fg0, fg1) not in self.caches["bandpower"]:
                            mean = bX.mean(dim=[0,2], keepdim=True)
                            std = bX.std(dim=[0,2], keepdim=True)
                            self.caches["bandpower"][(fg0, fg1)] = (mean, std)
                        mean, std = self.caches["bandpower"][(fg0, fg1)]
                        bX = bX - mean
                        bX = bX / std

                    cur_cache[(fg0, fg1)] = bX
                _subbands = _subbands[MAX_PROCESSED_SUBBANDS:]
                _gbands = _gbands[MAX_PROCESSED_SUBBANDS:]


        if cache is not None:
            for (fg0, fg1), bX in cur_cache.items():
                cache.update((fg0, fg1), bX)

        covs = []
        for tg0, tg1, fg0, fg1 in tfgs:
            bX = cur_cache[(fg0, fg1)]
            p0 = tg0 * self.p_stride
            p1 = tg1 * self.p_stride
            if return_cov:
                cov = bX[..., p0:p1] @ bX[..., p0:p1].transpose(-1, -2) / ((p1 - p0) / self.fs)
            else:
                mask = torch.zeros(bX.shape[-1], device=bX.device)
                mask[p0:p1] = 1.
                cov = bX * mask
            covs.append(cov)

        covs = torch.stack(covs, dim=1)
        return covs

    @record_function("get_tfcov_mean")
    @calc_runtime("get_tfcov_mean")
    def get_tfcov_mean(self, tfgs, X=None, mode=None, use_cache=True):
        # X:  n_b x n_c x n_p
        # tfgs: n_g x [tg0, tg1, fg0, fg1]
        # Output: n_b x n_g x n_c x n_c
        if mode is None:
            mode = self.f_mode
        match mode:
            case "lasyexact":
                return self.get_tfcov_mean_lazyexact(tfgs, X=X, use_cache=use_cache)
            case _:
                raise Exception("Unknown mode")

    @staticmethod
    @record_function("csp_fit")
    def _csp_fit(m0, m1, k, tkreg=0.):
        return calc_CSP_filter(m0, m1, k=k, eps=tkreg, niter=20)

    @record_function("csp_fit_para")
    @calc_runtime("csp_fit_para")
    def _csp_fit_parallal(self, m0, m1, k, tkreg_list):
        # m0/m1: ... x n_c x n_c
        # Output: ... x n_tk x 2k
        ori_shape = list(m0.shape)
        n_c = ori_shape[-1]
        As = []
        Bs = []
        for tkreg in tkreg_list:
            if tkreg > 0:
                tkreg = torch.eye(m0.shape[-1], device=m0.device) * tkreg
            As.append(m0)
            Bs.append(m1 + tkreg)
            As.append(m1)
            Bs.append(m0 + tkreg)
        As = torch.stack(As, dim=-3)  # ... x (n_tk*2) x n_c x n_c
        Bs = torch.stack(Bs, dim=-3)
        ws = solve_GRQ(As, Bs, niter=20, k=k)  # ... x (n_tk*2) x k
        target_shape = ori_shape[:-2] + [len(tkreg_list), 2 * k, n_c]
        ws = ws.reshape(target_shape)
        return ws

    @record_function("csp_transform")
    @calc_runtime("csp_transform")
    def _csp_transform(self, ws, X):
        return apply_CSP_filter(ws, X, do_log=True, select_k=None)

    @staticmethod
    def _select_k(X, select_k=None):
        if select_k is not None:
            max_k = X.shape[-1]  # e.g.  8
            e1 = select_k  # e.g.  3
            s2 = max_k // 2  # e.g.  4
            e2 = s2 + select_k  # e.g.  7
            X = torch.cat([X[..., :e1], X[..., s2:e2]], dim=-1)
        return X

    @record_function("als_fit")
    @calc_runtime("als_fit")
    def _als_fit(self, X, Y, select_k=None):
        X = self._select_k(X, select_k)
        return calc_lda_filter(X, Y)

    @record_function("als_fit_parallal")
    @calc_runtime("als_fit_parallal")
    def _als_fit_parallal(self, Xs, Ys):
        return calc_lda_filter_parallal(Xs, Ys)

    @record_function("als_transform")
    @calc_runtime("als_transform")
    def _als_transform(self, wb, X, select_k=None):
        w, b = wb
        if select_k is not None:
            max_k = X.shape[-1]  # e.g.  8
            e1 = select_k  # e.g.  3
            s2 = max_k // 2  # e.g.  4
            e2 = s2 + select_k  # e.g.  7
            X = torch.cat([X[..., :e1], X[..., s2:e2]], dim=-1)
        return apply_lda_filter(X, w, b)

    @record_function("calc_AFR")
    @calc_runtime("calc_AFR")
    def calc_AFR_score(self, cv_setting: cvsetting_type, tfgs: tfglist_type, X, Y, ks, tkregs, mode=None,
                       use_cache=True, return_detail=False):
        # cv_settings:  (tinds, vinds)
        # ks: list of possible k
        # tkregs: list of possible tkregs
        tfcov = self.get_tfcov_mean(tfgs, X=X, mode=mode, use_cache=use_cache)  # n_b x n_g x n_c x n_c
        tinds, vinds = cv_setting
        max_k = max(ks)
        als = []
        ys = []

        # CV-CSP Parallal Preprocess
        m0s = []
        m1s = []
        for tind, vind in zip(tinds, vinds):
            tx = tfcov[tind]
            ty = Y[tind]
            m0 = tx[ty == 0].mean(dim=0)
            m1 = tx[ty == 1].mean(dim=0)  # n_g x n_c x n_c
            m0s.append(m0)
            m1s.append(m1)
        m0s = torch.stack(m0s)
        m1s = torch.stack(m1s)  # n_cv x n_g x n_c x n_c
        ws = self._csp_fit_parallal(m0s, m1s, max_k, tkregs)  # n_cv x n_g x n_tk x (2k)

        # CV-LDA Parallal Preprocess
        wb_per_k = {}
        for k in ks:
            Xs = []
            Ys = []
            for cvind, (tind, vind) in enumerate(zip(tinds, vinds)):
                tx = tfcov[tind]  # n_b x n_g x n_c x n_c
                ty = Y[tind]
                tfs = self._csp_transform(ws[cvind], tx[..., None, :, :])
                # fs:  n_b x n_g x n_tk x 2max_K
                Xs.append(self._select_k(tfs, k))
                Ys.append(ty)
            wb_per_k[k] = self._als_fit_parallal(Xs, Ys)

        for cvind, (tind, vind) in enumerate(zip(tinds, vinds)):
            vx = tfcov[vind]
            vy = Y[vind]
            vfs = self._csp_transform(ws[cvind], vx[..., None, :, :])
            # fs:  n_b x n_g x n_tk x 2max_K
            # tos = []
            vos = []
            for k in ks:
                wb = (wb_per_k[k][0][cvind], wb_per_k[k][1][cvind])
                vo = self._als_transform(wb, vfs, k)
                vos.append(vo)
            vos = torch.stack(vos, dim=-1)
            # vos: n_b x n_g x n_tk x n_k
            als.append(vos)
            ys.append(vy)

        if self.criterion=="AFR":
            als = torch.cat(als, dim=0)
            ys = torch.cat(ys, dim=0)
            afr = calc_fr(als[ys == 1], als[ys == 0], oneway=True, axis=0)
            score = afr
        if self.criterion=="MFR":
            frs = []
            for xx, yy in zip(als, ys):
                fr = calc_fr(xx[yy==1], xx[yy==0], oneway=True, axis=0)
                frs.append(fr)
            frs = torch.stack(frs).mean(dim=0)
            score = frs
        if self.criterion=="CVACC":
            accs = []
            for xx,yy in zip(als, ys):
                accs.append((xx>0).float().eq(yy[:, None, None, None]).float().mean(dim=0))
            accs = torch.stack(accs).mean(dim=0)
            score = accs

        if return_detail:
            return score
        else:
            return score.amax(dim=[1, 2])

    @record_function("calc_overlap")
    @calc_runtime("calc_overlap")
    def calc_area_overlap(self, old_tfs: tflist_type, new_tfs: tflist_type, overlap_gamma=2., overlap_mode="max"):
        ratios = []
        for tfs in old_tfs:
            at0, at1, af0, af1 = tfs
            ratio = []
            for bt0, bt1, bf0, bf1 in new_tfs:
                dx = min(at1, bt1) - max(at0, bt0)
                dy = min(af1, bf1) - max(af0, bf0)
                if (dx >= 0) and (dy > 0):
                    overlap = dx * dy
                    sa = (at1 - at0) * (af1 - af0)
                    sb = (bt1 - bt0) * (bf1 - bf0)
                    if overlap_mode == "max":
                        ratio.append(overlap / max(sa, sb))
                    else:
                        assert overlap_mode == "min"
                        ratio.append(overlap / min(sa, sb))
                else:
                    ratio.append(0.)
            ratio = torch.tensor(ratio)
            ratio **= overlap_gamma
            ratios.append(ratio)

        return torch.stack(ratios, dim=0).amax(dim=0)  # len(old_tfs) x n_g -> n_g

    @record_function("calc_score")
    @calc_runtime("calc_score")
    def _calc_score_with_cache(self, tfgs: tfglist_type, old_tfgs: tfglist_type):
        cache_afr = self.caches['afr']
        assert "cv_settings" in self.caches
        cv_settings = self.caches["cv_settings"]
        X = self.caches["train_X"]
        Y = self.caches["train_Y"]
        f_mode = self.f_mode
        checked_ks = self.checked_ks
        checked_tkregs = self.checked_tkregs
        o_g = self.overlap_gamma
        o_m = self.overlap_mode

        to_be_added_tfgs = []
        for tfg in tfgs:
            if tfg not in cache_afr:
                to_be_added_tfgs.append(tfg)

        if len(to_be_added_tfgs) > 0:
            afrs = self.calc_AFR_score(cv_settings, to_be_added_tfgs, X, Y, checked_ks, checked_tkregs, f_mode,
                                       use_cache=self._use_cache_in_training, return_detail=True)
            for ii in range(len(to_be_added_tfgs)):
                cache_afr[to_be_added_tfgs[ii]] = afrs[ii].detach().cpu()

        scores = torch.tensor([cache_afr[tfg].amax(dim=[-1, -2]).item() for tfg in tfgs])
        if len(old_tfgs) > 0:
            overlapped_ratio = self.calc_area_overlap(old_tfgs, tfgs, o_g, o_m)
        else:
            overlapped_ratio = 0.
        return scores * (1 - overlapped_ratio)

    @staticmethod
    def _make_HC_candidates(g0: int, g1: int, max_step: int, step_length: int,
                            min_g: int, max_g: int, min_width: int, max_width: int):
        candidates = set()
        # 平移生成候选
        for d in range(-max_step, max_step + 1):
            new_g0 = g0 + d * step_length
            new_g1 = g1 + d * step_length
            if min_g <= new_g0 <= max_g and min_g <= new_g1 <= max_g and min_width <= (new_g1 - new_g0) <= max_width:
                candidates.add((new_g0, new_g1))

        # 缩放生成：调整 g0（保持 g1 不变）
        for d in range(-max_step, max_step + 1):
            new_g0 = g0 + d * step_length
            new_g1 = g1
            if min_g <= new_g0 <= max_g and min_g <= new_g1 <= max_g and min_width <= (new_g1 - new_g0) <= max_width:
                candidates.add((new_g0, new_g1))

        # 缩放生成：调整 g1（保持 g0 不变）
        for d in range(-max_step, max_step + 1):
            new_g0 = g0
            new_g1 = g1 + d * step_length
            if min_g <= new_g0 <= max_g and min_g <= new_g1 <= max_g and min_width <= (new_g1 - new_g0) <= max_width:
                candidates.add((new_g0, new_g1))

        # 返回所有候选对的列表
        return list(candidates)

    def time2grid(self, tm):
        grid = tm * self.fs // self.p_stride
        return int(grid)

    def grid2time(self, grid):
        tm = grid * self.p_stride / self.fs
        return tm

    def freq2grid(self, freq):
        grid = (freq - self.f_min_hz) // self.f_stride_hz
        return int(grid)

    def grid2freq(self, grid):
        freq = grid * self.f_stride_hz + self.f_min_hz
        return freq

    @record_function("CD")
    @calc_runtime("CD")
    def CD_search(self,
                  init_tfgs: tfg_type,
                  old_tfgs: tfglist_type,
                  time_steplength: int,
                  freq_steplength: int, ):

        best_tfg = tg0, tg1, fg0, fg1 = init_tfgs
        time_min_width = self.min_windowlength_s * self.fs // self.p_stride
        time_max_width = self.max_windowlength_s * self.fs // self.p_stride
        freq_min_width = self.min_bandwidth_hz // self.f_stride_hz
        freq_max_width = self.max_bandwidth_hz // self.f_stride_hz
        time_min = self.time2grid(self.t_min_s)
        time_max = self.time2grid(self.t_max_s)
        freq_min = self.freq2grid(self.f_min_hz)
        freq_max = self.freq2grid(self.f_max_hz)
        time_max_step = self.time_max_step
        freq_max_step = self.freq_max_step
        max_iter = self.cd_maxiter

        for n_iter in range(max_iter):
            # Temporal Search
            gs = self._make_HC_candidates(tg0, tg1, time_max_step, time_steplength,
                                          time_min, time_max, time_min_width, time_max_width)
            temporal_tfgs = [(g0, g1, fg0, fg1) for g0, g1 in gs]
            temporal_scores = self._calc_score_with_cache(temporal_tfgs, old_tfgs)  # CPU Float Tensor

            # Spectral Search
            gs = self._make_HC_candidates(fg0, fg1, freq_max_step, freq_steplength,
                                          freq_min, freq_max, freq_min_width, freq_max_width)

            spectral_tfgs = [(tg0, tg1, g0, g1) for g0, g1 in gs]
            spectral_scores = self._calc_score_with_cache(spectral_tfgs, old_tfgs)  # CPU Float Tensor

            # Find the best

            best_tind = temporal_scores.argmax().item() if len(temporal_scores) > 0 else None
            best_find = spectral_scores.argmax().item() if len(spectral_scores) > 0 else None
            if best_tind is None:
                best_tfg = spectral_tfgs[best_find]
            elif best_find is None:
                best_tfg = temporal_tfgs[best_tind]
            elif temporal_scores[best_tind] > spectral_scores[best_find]:
                # best_score = temporal_scores[best_tind]
                best_tfg = temporal_tfgs[best_tind]
            else:
                # best_score = spectral_scores[best_find]
                best_tfg = spectral_tfgs[best_find]
            if best_tfg == (tg0, tg1, fg0, fg1):
                return best_tfg
            else:
                tg0, tg1, fg0, fg1 = best_tfg

        return best_tfg

    @record_function("VSCD")
    def VSCD_search(self, init_tfgs: tfg_type, old_tfgs: tfglist_type):
        best_tfg = init_tfgs
        for tsm, fsm in zip(self.time_step_multiple, self.freq_step_multiple):
            best_tfg = self.CD_search(best_tfg, old_tfgs, tsm, fsm)

        return best_tfg

    def random_init_search(self, old_tfgs: tfglist_type, verbose=True):
        random_init_timewindow_range_s = self.random_init_timewindow_range_s
        random_init_timewindow_per_band = self.random_init_timewindow_per_band
        twl, twr = random_init_timewindow_range_s
        max_fg = self.freq2grid(self.f_max_hz)
        min_fg = self.freq2grid(self.f_min_hz)
        min_tg = self.time2grid(self.t_min_s)
        max_tg = self.time2grid(self.t_max_s)
        fgm = self.freq_step_multiple[0]
        tgm = self.time_step_multiple[0]
        min_winlength_g = self.time2grid(twl)
        max_winlength_g = self.time2grid(twr)

        possible_tg0 = [tg for tg in range(min_tg, max_tg + 1, tgm)]

        valid_pairs = []

        for tg0 in possible_tg0:
            min_tg1 = tg0 + min_winlength_g
            max_tg1 = min(tg0 + max_winlength_g, max_tg)

            # 生成符合条件的 tg1
            possible_tg1 = [tg for tg in range(min_tg1, max_tg1 + 1, tgm)]

            for tg1 in possible_tg1:
                valid_pairs.append((tg0, tg1))

        fg0 = min_fg
        fg1 = fg0 + fgm
        last_best_score = -9999
        last_best_tfg = None
        while fg1 <= max_fg:
            for _ in range(random_init_timewindow_per_band):
                tg0, tg1 = random.choice(valid_pairs)
                best_tfg = self.VSCD_search((tg0, tg1, fg0, fg1), old_tfgs)
                best_score = self._calc_score_with_cache([best_tfg], old_tfgs).item()
                if verbose:
                    print(".", end="", flush=True)
                if best_score > last_best_score:
                    last_best_score = best_score
                    last_best_tfg = best_tfg
                # raise ValueError("")

            fg0 = fg0 + fgm
            fg1 = fg0 + fgm

        return last_best_tfg, last_best_score

    def clear_cache(self):
        # Create CV Settings
        self.caches = {
            "afr": {},
            "bandpower": {},
        }

    def fit_binary(self, TX, TY, max_windows=30, seed=0, verbose=True):
        self.clear_cache()
        total_cv = self.als_CV
        if self.manual_cv_settings is None:
            cv_settings = self._create_cv_settings(len(TX), TY, total_cv, seed)
        else:
            if verbose:
                print("Use manual CV settings")
            cv_settings = self.manual_cv_settings
        self.caches["cv_settings"] = cv_settings
        self.caches["train_X"] = TX
        self.caches["train_Y"] = TY
        self.runtime = {}
        self._last_count_filter = 0
        self.count_afr = []
        self.count_filter = []
        self.runtime_shots = []
        manual_seed(seed)
        details = []
        tfgs = []
        for ii in range(max_windows):
            if verbose:
                print(f"Searching window No. {ii + 1}: ", end="")
            best_tfgs, best_score = self.random_init_search(tfgs, verbose=verbose)
            tfgs.append(best_tfgs)
            tg0, tg1, fg0, fg1 = best_tfgs
            t0 = self.grid2time(tg0)
            t1 = self.grid2time(tg1)
            f0 = self.grid2freq(fg0)
            f1 = self.grid2freq(fg1)
            if verbose:
                print(f"{t0:.1f}-{t1:.1f}s {f0:.1f}-{f1:.1f}Hz Score {best_score:.3f}")
            self.count_filter.append(self._last_count_filter)
            self.count_afr.append(len(self.caches["afr"]))
            self.runtime_shots.append(self.runtime.copy())
            cv_settings = self.caches["cv_settings"]
            X = self.caches["train_X"]
            Y = self.caches["train_Y"]
            ks = self.checked_ks
            tkregs = self.checked_tkregs
            details_afrs = self.calc_AFR_score(cv_settings, [best_tfgs], X, Y, ks, tkregs, "lasyexact", True, True)
            afr = details_afrs[0]
            tkreg_ind, k_ind = afr.amax().eq(afr).argwhere()[0]
            best_tk = tkregs[tkreg_ind]
            best_k = ks[k_ind]
            details.append({
                "tfg": best_tfgs,
                "t0": t0,
                "t1": t1,
                "f0": f0,
                "f1": f1,
                "afr": best_score,
                "count_filter": self.count_filter[-1],
                "count_afr": self.count_afr[-1],
                "runtime": self.runtime_shots[-1],
                "tkreg": best_tk,
                "k": best_k,
            })

        return details

    def extract_signal_binary(self, state, TX, do_sum=True):
        tfgs = [ws['tfg'] for ws in state]
        sig = self.get_tfcov_mean_lazyexact(tfgs, TX, False, False)
        if do_sum:
            sig = sig.sum(dim=1)
        return sig

    def _select_binary_classes(self, X, Y, c0, c1):
        X, Y = self._select_classes(X, Y, [c0, c1])
        return X, Y

    @staticmethod
    def _normalize_mc_mode(mc_mode):
        mc_mode = mc_mode.lower()
        if mc_mode not in ["ovo", "ovr"]:
            raise ValueError(f"Unsupported multiclass mode: {mc_mode}")
        return mc_mode

    def _get_class_list(self, Y, class_list=None):
        if class_list is None:
            max_cls = int(max(Y) + 1)
            class_list = list(range(max_cls))
        return list(class_list)

    def _get_multiclass_tasks(self, class_list, mc_mode="ovo"):
        mc_mode = self._normalize_mc_mode(mc_mode)
        if mc_mode == "ovo":
            return [("ovo", (c0, c1)) for c0, c1 in itertools.combinations(class_list, 2)]
        return [("ovr", cc) for cc in class_list]

    @staticmethod
    def _task_to_key(task):
        mode, spec = task
        if mode == "ovo":
            return tuple(spec)
        if mode == "ovr":
            return ("ovr", spec)
        raise ValueError(f"Unsupported task mode: {mode}")

    def _select_task_classes(self, X, Y, task):
        mode, spec = task
        if mode == "ovo":
            c0, c1 = spec
            return self._select_binary_classes(X, Y, c0, c1)
        if mode == "ovr":
            positive_class = spec
            X = X.clone()
            Y = Y.clone()
            Y = (Y == positive_class).long()
            return X, Y
        raise ValueError(f"Unsupported task mode: {mode}")

    def _select_classes(self, X, Y, class_list):
        X = X.clone()
        Y = Y.clone()
        mask = torch.full_like(Y, False, dtype=torch.bool)
        cmasks = []
        for cc in class_list:
            cmask = Y == cc
            mask = mask | cmask
            cmasks.append(cmask)
        for cind, cmask in enumerate(cmasks):
            Y[cmask] = cind
        X = X[mask]
        Y = Y[mask]
        return X, Y

    def fit(self, TX, TY, max_windows=30, seed=0, verbose=True, mc_mode="ovo", class_list=None):
        """
        Fit the PTFCA window search over a multiclass problem.

        mc_mode:
            "ovo": one binary task per class pair, keys are (c0, c1) tuples.
            "ovr": one binary task per class (class vs. the rest), keys are ("ovr", c) tuples.
        class_list:
            [..., ci, ...] Pre-selected (original) class labels to fit on.
                - Labels are remapped to 0, 1, ... in the returned details_dict.
                None: class_list = [0, 1, ..., TY.amax()]

        The default mc_mode="ovo" with class_list=None reproduces the binary behaviour
        of previous versions: keys (c0, c1) valid for the fitted classes.
        """
        class_list = self._get_class_list(TY, class_list)
        selected_class_list = list(class_list)
        TX, TY = self._select_classes(TX, TY, class_list)
        class_list = list(range(len(class_list)))
        tasks = self._get_multiclass_tasks(class_list, mc_mode)
        details_dict = {
            "info": {
                "max_windows": max_windows,
                "seed": seed,
                "mc_mode": self._normalize_mc_mode(mc_mode),
                "class_list": class_list,
                "selected_class_list": selected_class_list,
            }
        }
        for task in tasks:
            key = self._task_to_key(task)
            if verbose:
                print("Fitting task:", key)
            _X, _Y = self._select_task_classes(TX, TY, task)
            details_dict[key] = self.fit_binary(_X, _Y, max_windows, seed, verbose)
        return details_dict

    def extract_signal(self, OVO_state, TX, do_sum=True, max_wins=None):
        tfgs = []
        for k, state in OVO_state.items():
            if k!="info":
                tfg = [ws['tfg'] for ws in state]
                if max_wins is not None:
                    tfg = tfg[:max_wins]
                tfgs += tfg
        sig = self.get_tfcov_mean_lazyexact(tfgs, TX, False, False)
        if do_sum:
            sig = sig.sum(dim=1)
        return sig

    def binary_CSP_LDA_fit(self, details, TX, TY, class_ind=None):
        """
        Fit CSP-LDA transformer in binary case.
            details = self.fit_binary(...)
            class_ind:
                None:  Assuming TY in  [0,1]
                (c0, c1):  Assuming TY has elements > 1, only class c0 and c1 will be used.
        """
        if class_ind is not None:
            # Tuple: cls0, cls1
            c0 = class_ind[0]
            c1 = class_ind[1]
            TX, TY = self._select_binary_classes(TX, TY, c0, c1)
        tfgs = [d['tfg'] for d in details]
        TX = self.get_tfcov_mean_lazyexact(tfgs, TX, use_cache=False).cpu()
        TY = TY.cpu()
        states = []
        for ind, detail in enumerate(details):
            state = {}
            tkreg = detail["tkreg"]
            k = detail["k"]
            state['ws'] = ws = self._csp_fit(TX[TY == 0, ind].mean(dim=0), TX[TY == 1, ind].mean(dim=0), k, tkreg)
            TF = self._csp_transform(ws, TX[:, ind])
            state["wb"] = self._als_fit(TF, TY)
            state['tfg'] = detail["tfg"]
            states.append(state)
        return states

    def binary_CSP_LDA_transform(self, states, EX, target="als"):
        """
        states = self.binary_CSP_LDA_fit(...)
        EX: n_b x n_c x n_p
        target in ["als", "csp"]
        Output:
            List[n_b x 2k]  # target=='csp', k can vary.
            List[n_b x 1]   # target=='als'
            # len(List)=len(states)
        """
        assert target in ['als', 'csp']
        tfgs = [d['tfg'] for d in states]
        EX = self.get_tfcov_mean_lazyexact(tfgs, EX, use_cache=False).cpu()
        EOs = []
        for ind, state in enumerate(states):
            ws = state['ws']
            wb = state['wb']
            EF = self._csp_transform(ws, EX[:, ind])
            if target == 'csp':
                EOs.append(EF)
                continue
            EO = self._als_transform(wb, EF)
            EOs.append(EO[..., None])

        return EOs

    def CSP_LDA_fit(self, details_dict, TX, TY, class_list=None, mc_mode=None):
        """
        A multi-class solution for calculating ALS.The CSP-LDA is fit across all O-V-O pairs;
            details_dict = self.fit(...)
            TX, TY: Should be the same as that used in self.fit(...)
            class_list: [..., ci, ...] Pre-selected classes.
                - ci should in details_dict and TY.
                None:  class_list = [0, 1, ..., TY.amax()]
            mc_mode: "ovo" (class pairs) or "ovr" (one-vs-rest).
                None: taken from details_dict["info"], defaulting to "ovo".

        return: List[List[state per window]]
        """
        if mc_mode is None:
            mc_mode = details_dict.get("info", {}).get("mc_mode", "ovo")
        class_list = self._get_class_list(TY, class_list)
        cTX, cTY = self._select_classes(TX, TY, class_list)
        class_list = list(range(len(class_list)))
        tasks = self._get_multiclass_tasks(class_list, mc_mode)
        states_mc = []
        for task in tasks:
            key = self._task_to_key(task)
            details = details_dict[key]
            mode, spec = task
            if mode == "ovo":
                states = self.binary_CSP_LDA_fit(details, cTX, cTY, spec)
            else:
                _X, _Y = self._select_task_classes(cTX, cTY, task)
                states = self.binary_CSP_LDA_fit(details, _X, _Y)
            states_mc.append(states)

        return states_mc

    def CSP_LDA_transform(self, states_OVO, EX, target="als"):
        """
        states_OVO = self.CSP_LDA_fit(...)
        Return:  List[n_b x (n_ovo * ?)]  ? = 1 or 2k (k can vary)
        """
        EO_OVO = []
        for states in states_OVO:
            EO_OVO.append(self.binary_CSP_LDA_transform(states, EX, target))
        n_g = len(states_OVO[0])
        EO_OVO = [torch.cat([EO_OVO[jj][ii] for jj in range(len(EO_OVO))], dim=1) for ii in range(n_g)]
        return EO_OVO

    @staticmethod
    def svm_fit(TF, TY, win=None, C=0.0002, svm_mc_mode="ovr", seed=None, pca=128):
        if win is None:
            win = len(TF)
        TY = to_numpy(TY)
        TF = to_numpy(torch.cat(TF[:win], dim=-1))
        pipeline = [("scaler",StandardScaler())]
        if pca is None or TF.shape[-1]<=pca:
            pass
        else:
            pipeline.append(("pca",PCA(pca, random_state=seed)))
        pipeline.append(("clf",LinearSVC(C=C, multi_class=svm_mc_mode, random_state=seed)))
        pipeline = Pipeline(pipeline)
        pipeline.fit(TF, TY)
        return pipeline

    @staticmethod
    def svm_predict(svm, EF, win=None):
        if win is None:
            win = len(EF)
        EF = to_numpy(torch.cat(EF[:win], dim=-1))
        return svm.predict(EF)

    @staticmethod
    def svm_score(svm, EF, EY, win=None):
        if win is None:
            win = len(EF)
        EF = to_numpy(torch.cat(EF[:win], dim=-1))
        EY = to_numpy(EY)
        return svm.score(EF, EY)

    def NN_fit(self, details_dict, TX, TY, batch_size=32, n_s=32, dropout=0., max_clfnorm=0.5,
               max_epoch=2000, lr0=0.0001, lr1=0.0001, seed=None, class_list=None, verbose=True,
               mc_mode=None):

        if mc_mode is None:
            mc_mode = details_dict.get("info", {}).get("mc_mode", "ovo")
        class_list = self._get_class_list(TY, class_list)
        selected_class_list = list(class_list)

        cTX, cTY = self._select_classes(TX, TY, class_list)
        class_list = list(range(len(class_list)))
        tasks = self._get_multiclass_tasks(class_list, mc_mode)
        tfgs_list = []
        TC_OVO = []
        for task in tasks:
            details = details_dict[self._task_to_key(task)]
            tfgs = [d['tfg'] for d in details]
            tfgs_list.append(tfgs)
            TC_OVO.append(self.get_tfcov_mean_lazyexact(tfgs, cTX, use_cache=False))
        TC_OVO = torch.stack(TC_OVO, dim=2)  # n_b x n_g x n_ovo x n_c x n_c
        _, n_g, n_ovo, n_c, _ = TC_OVO.shape
        TDL = TensorLoader(TC_OVO, cTY, batch_size, True)

        # Trace Normalized
        eye = torch.eye(TDL.X.shape[-1], device=TDL.X.device)
        mean_trace = (TDL.X * eye).sum(dim=[-1, -2], keepdim=True).mean(dim=0)
        TDL.X = TDL.X / mean_trace
        if seed is not None:
            manual_seed(seed)

        device = cTX.device
        if device.type == "cuda":
            TDL.cuda(device)
        else:
            TDL.cpu()
        net_para = {
            "n_g": n_g,
            "n_cp": n_ovo,
            "n_c": n_c,
            "n_s": n_s,
            "dropout": dropout,
            "n_class": len(class_list),
            "max_clfnorm": max_clfnorm,
            "max_spnorm": 9999999.,
        }
        net = MulticompNet(**net_para).to(device)

        state_dict_list = []
        if not isinstance(max_epoch, Sequence):
            epoch_breakpoints = [max_epoch]
        else:
            epoch_breakpoints = max_epoch
            max_epoch = max(epoch_breakpoints)

        def _save_net_at_certain_epoch(env):
            # env = {"epoch":epoch, "max_epoch":max_epoch, "model": model}
            if env['epoch'] in epoch_breakpoints:
                state_dict_list.append(deepcopy(net.state_dict()))

        fit_ce_loss_ex(net, TDL, max_epoch, lr0, lr1, 10, None, eval_update=50,
                       acc_aggfunc=1, prehook=_save_net_at_certain_epoch, verbose=verbose)

        state_dict_list.append(deepcopy(net.state_dict()))
        states = {
            "state_dict": state_dict_list,
            "epoch": epoch_breakpoints,
            "mean_trace": mean_trace,
            "net_para": net_para,
            "train_info": {
                "seed": seed,
                "lr0": lr0,
                "lr1": lr1,
            },
            "class_list": class_list,
            "selected_class_list": selected_class_list,
            "mc_mode": self._normalize_mc_mode(mc_mode),
            "tfgs_list": tfgs_list,
        }
        return states

    def NN_transform(self, states, EX, target="logvar"):
        """
        states = NN_fit(...)
        Transform test signals to logvar features / logits
            EX: n_b x n_c x n_p
            Output:
                List[n_b x ?]  # if len(epoch_breakpoints)==1
                List[List[n_b x ?]]  # For each epoch_breakpoints
                # ? == n_s*n_ovo  # target=='logvar'
                # ? == n_class  # target=='logit'
        """
        net_para = states['net_para']
        EC_OVO = []
        for tfgs in states["tfgs_list"]:
            EC_OVO.append(self.get_tfcov_mean_lazyexact(tfgs, EX, use_cache=False))
        EC_OVO = torch.stack(EC_OVO, dim=2)
        if not "--cpu" in sys.argv:
            EC_OVO = EC_OVO.cuda()
        mean_trace = states['mean_trace']
        EC_OVO = EC_OVO / mean_trace

        net = MulticompNet(**net_para)
        if not "--cpu" in sys.argv:
            net = net.cuda()
        net.eval()

        with torch.no_grad():
            output_list = []
            for state_dict in states['state_dict']:
                net.load_state_dict(state_dict)
                if target == 'logit':
                    out = batch_process(net.forward, EC_OVO, 32)
                elif target == 'logvar':
                    out = batch_process(net.forward_logvar, EC_OVO, 32)
                else:
                    raise Exception("?")
                # out: n_b x n_g x ?
                out = [out[:, ii] for ii in range(out.shape[1])]
                output_list.append(out)

        if len(output_list) == 1:
            output_list = output_list[0]

        return output_list


# %% Visualized Tools
def get_tf_region_masks(ptfca_state, t0, t1, f0, f1, t_num, f_num, weight=None):
    t_ticks = torch.linspace(t0, t1, t_num)
    f_ticks = torch.linspace(f0, f1, f_num)
    n_g = len(ptfca_state)

    mask = torch.zeros(n_g, f_num, t_num)
    if weight is None:
        weight = torch.ones(n_g)
    for ii in range(n_g):
        wt0 = ptfca_state[ii]['t0']
        wt1 = ptfca_state[ii]['t1']
        wf0 = ptfca_state[ii]['f0']
        wf1 = ptfca_state[ii]['f1']
        f_mask = (f_ticks >= wf0) & (f_ticks <= wf1)
        t_mask = (t_ticks >= wt0) & (t_ticks <= wt1)
        ft_mask = f_mask.float()[:, None] * t_mask.float()[None, :]
        mask[ii] += ft_mask*weight[ii]
    return mask

def plot_tf_box_with_afr(ptfca_state, t0, t1, f0, f1, mint=None, maxt=None, minf=None, maxf=None,
                         cmap="jet", afr_top=5, ecolors=None, linewidths=None, alphas=None, ax=None):
    df = pd.DataFrame(ptfca_state, columns=['t0', 't1', 'f0', 'f1', 'afr'])
    df = df.sort_values("afr", ascending=False)
    ax: plt.Axes
    ax.set_ylim(f0, f1)
    ax.set_xlim(t0, t1)
    cmap = plt.get_cmap(cmap)
    iterrows = list(df.iterrows())
    for ind, item in iterrows[::-1]:
        t0, t1, f0, f1, afr = item
        linewidth = max(6 - ind, 2)
        alpha = max(10 - ind, 2) / 10.
        ecolor = cmap(min(afr, afr_top) / afr_top)
        rect = patches.Rectangle((t0, f0), t1 - t0, f1 - f0,
                                 linewidth=linewidth if linewidths is None else linewidths[ind],
                                 edgecolor=ecolor if ecolors is None else ecolors[ind],
                                 alpha=alpha if alphas is None else alphas[ind],
                                 facecolor='none')
        ax.add_patch(rect)
    if mint is not None:
        ax.axvline(mint, color="grey", linestyle="--")
    if maxt is not None:
        ax.axvline(maxt, color="grey", linestyle="--")
    if minf is not None:
        ax.axhline(minf, color="grey", linestyle="--")
    if maxf is not None:
        ax.axhline(maxf, color="grey", linestyle="--")

