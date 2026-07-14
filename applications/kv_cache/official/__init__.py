"""Official-paradigm KV-cache baselines (prefill-time compression).

Faithful ports of H2O / SnapKV / PyramidKV that compress once at end of prefill,
installed via a forward-hook monkeypatch. See clusters.py and monkeypatch.py.
"""

from .clusters import SnapKVCluster, H2OKVCluster, PyramidKVCluster
from .monkeypatch import PrefillCompressor


def make_cluster_factory(method, max_capacity, window_size=32,
                         kernel_size=5, pooling="avgpool", beta=20):
    """Return a `make_cluster(layer_idx, num_layers)` builder for the given method.

    method: "h2o" | "snapkv" | "pyramidkv" (official prefill-compression paradigm).
    max_capacity: target retained prompt length (per layer for h2o/snapkv;
        the pyramid's centre for pyramidkv).
    """
    m = method.lower().removesuffix("_official")

    def factory(layer_idx, num_layers):
        if m == "snapkv":
            return SnapKVCluster(window_size=window_size,
                                 max_capacity_prompt=max_capacity,
                                 kernel_size=kernel_size, pooling=pooling)
        if m == "h2o":
            return H2OKVCluster(window_size=window_size,
                                max_capacity_prompt=max_capacity)
        if m == "pyramidkv":
            return PyramidKVCluster(num_hidden_layers=num_layers,
                                    window_size=window_size,
                                    max_capacity_prompt=max_capacity,
                                    kernel_size=kernel_size, pooling=pooling,
                                    beta=beta, layer_idx=layer_idx)
        raise ValueError(f"unknown official method: {method}")

    return factory


__all__ = ["SnapKVCluster", "H2OKVCluster", "PyramidKVCluster",
           "PrefillCompressor", "make_cluster_factory"]
