"""QORE-RAG package with lazy loading for lightweight diagnostics."""


def select_passages(*args, **kwargs):
    """Preserve the public API without importing heavy QORE dependencies."""

    from .selector import select_passages as _select_passages

    return _select_passages(*args, **kwargs)
