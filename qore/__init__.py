"""QORE: Quantum-Optimized Context Reduction for LLMs."""

from .qubo import build_qubo_matrix, energy
from .signals import cosine_redundancy, rbf_redundancy, normalize
from .block_decompose import decompose, recompose
from .solvers import solve
from .kernels import quantum_kernel

__version__ = "0.1.0"
