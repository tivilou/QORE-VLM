"""Question-centred, cross-fitted density-kernel readout on compact toy features.

Quantum kernels are classically simulated. This is a new readout experiment,
not a claim that kernel methods or density discrimination are new. All feature
fitting and class operators use training inputs/labels only.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from applications.rag.qkernel_v001 import states
from applications.rag.qvalue_portable_v001 import need

ARMS = ('RBF_FULL', 'RBF_ANGLES', 'MLP_KERNEL', 'POLY2', 'PRODUCT', 'DEPHASED', 'QUANTUM')


def prepared(train, heldout):
    """Full-50 within-question centering; PCA/scale fitted on training only."""
    need(train.shape[:2] == (80, 50) and heldout.shape[:2] == (20, 50), 'fixed feature roles')
    a = train-train.mean(1, keepdim=True)
    b = heldout-heldout.mean(1, keepdim=True)
    flat = a.flatten(0, 1)
    mean = flat.mean(0)
    centered = flat-mean
    covariance = centered.T@centered/len(centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    basis = eigenvectors[:, -12:].flip(1).float()
    # Eigenvector sign is fixed by the largest absolute coordinate, independent
    # of labels, model scores and heldout inputs.
    pivots = basis.abs().argmax(0)
    signs = basis[pivots, torch.arange(12, device=basis.device)].sign()
    basis = basis*signs
    scale = eigenvalues[-12:].flip(0).sqrt().float()
    need(bool((scale > 1e-6).all()), 'twelve informative training PCA directions')
    full = torch.cat((a, b)).flatten(0, 1)-mean
    coordinates = (full@basis)/scale
    unit = F.normalize(full, dim=-1)
    return dict(coordinates=coordinates, unit=unit, basis=basis, scale=scale, mean=mean,
                dimensions=int(train.shape[-1]), fit_questions=80)


def bandwidth(x):
    sample = x[:4000:16]  # Fixed input-only, all 80 questions represented.
    distances = torch.cdist(sample, sample).square()
    mask = ~torch.eye(len(sample), dtype=torch.bool, device=x.device)
    result = distances[mask].median()
    need(bool(torch.isfinite(result)) and float(result) > 1e-8, 'positive training median bandwidth')
    return result


def kernel_features(prep, arm, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    rotation, _ = torch.linalg.qr(torch.randn(12, 12, generator=generator))
    angles = math.pi*torch.tanh(prep['coordinates']@rotation.to(prep['coordinates'].device))
    if arm == 'RBF_FULL':
        x = prep['unit']
        return x, bandwidth(x), False
    if arm == 'RBF_ANGLES':
        return angles, bandwidth(angles), False
    if arm == 'POLY2':
        return prep['unit'], None, False
    if arm == 'MLP_KERNEL':
        g = torch.Generator(device='cpu').manual_seed(seed)
        w = (torch.randn(16, 12, generator=g)/math.sqrt(12)).to(angles.device)
        bias = (.1*torch.randn(16, generator=g)).to(angles.device)
        u = F.normalize(F.linear(angles, w, bias).tanh(), dim=-1)
        # Squared real overlap, same rank-one density feature dimension16^2.
        return u, None, True
    u = states(angles, arm != 'PRODUCT')
    need(float((u.abs().square().sum(-1)-1).abs().max()) < 1e-5, 'actual unit quantum states')
    if arm == 'DEPHASED':
        # Normalization controls purity-dependent attenuation. This is a
        # normalized HS-kernel ablation, not a probability/Born measurement.
        return F.normalize(u.abs().square(), dim=-1), None, False
    return u, None, True


def features(prep, labels, arm, seed):
    need(arm in ARMS and labels.shape == (80, 50), 'fixed arm/training targets')
    x, width, squared = kernel_features(prep, arm, seed)
    if arm.startswith('RBF_'):
        gram = torch.exp(-torch.cdist(x, x[:4000]).square()/width)
    else:
        gram = x.conj()@x[:4000].T
        gram = gram.abs().square() if squared else gram.real
        if arm == 'POLY2':
            gram = ((gram+1)/2).square()
    need(bool(torch.isfinite(gram).all()), 'finite complete kernel')
    y = labels.flatten()
    # Each toy question is one distinct original component. The stable
    # pre-score component order fixes five cross-fitting folds.
    fold = torch.arange(80, device=x.device).repeat_interleave(50) % 5
    training = torch.empty(4000, device=x.device)
    memberships = []
    for f in range(5):
        permitted = fold != f
        positive, negative = permitted & (y == 1), permitted & (y == 0)
        need(bool(positive.any()) and bool(negative.any()), 'both classes outside query component fold')
        weights = positive.float()/positive.sum()-negative.float()/negative.sum()
        query = fold == f
        training[query] = gram[:4000][query]@weights
        memberships.append(dict(fold=f, reference_questions=64,
                                positive=int(positive.sum()), negative=int(negative.sum())))
    weights = (y == 1).float()/(y == 1).sum()-(y == 0).float()/(y == 0).sum()
    heldout = gram[4000:]@weights
    mean, scale = training.mean(), training.std(unbiased=False)
    need(float(scale) > 1e-8, 'nonconstant cross-fitted training contrast')
    result = torch.cat((training, heldout))
    result = ((result-mean)/scale).reshape(100, 50)
    # Independent density/inner-product identity on a fixed prefix, not the
    # same GEMM repeated. Squared overlap equals a Hilbert-Schmidt inner product.
    oracle_error = None
    if squared:
        a, b = x[:4], x[50:54]
        rho = a[:, :, None]*a.conj()[:, None, :]
        sigma = b[:, :, None]*b.conj()[:, None, :]
        oracle = torch.einsum('aij,bji->ab', rho, sigma).real
        expected = (a.conj()@b.T).abs().square()
        oracle_error = float((oracle-expected).abs().max())
        need(oracle_error < 1e-5, 'independent density-kernel identity')
    return result.detach(), dict(arm=arm, seed=seed, folds=memberships,
        heldout_reference_questions=80, density_oracle_error=oracle_error,
        training_feature_std=float(scale), fitted_encoding_parameters=0,
        mean=float(mean), bandwidth=None if width is None else float(width),
        feature_parameters='Frozen input-only PCA/encoding; learned native linear weight plus one scalar')
