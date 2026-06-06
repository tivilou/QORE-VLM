"""
VQC (Variational Quantum Circuit) encoder for unified token scoring.

A single parameterized circuit U(h; θ) maps token features to quantum states,
producing BOTH quality scores (measurements) and redundancy signals (fidelity)
from the same quantum representation.
"""

from .encoder import VQCEncoder
from .scorer import vqc_score
