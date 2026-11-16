"""Check local Python dependencies without importing models, reading data or networking."""
from __future__ import annotations
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = ('scripts/run_triviaqa_smallset.py', 'scripts/run_top8.py',
               'applications/rag/tests/test_top8_binary_numerics.py',
               'applications/rag/tests/test_top8_cache_numerics.py')
LOCAL_PACKAGES = {'applications', 'scripts'}


def dependency_files(root=ROOT, entries=ENTRYPOINTS):
    """Follow local imports, including deferred runner imports and namespace packages."""
    root = Path(root).resolve()
    pending = list(entries)
    seen, missing = set(), set()

    def resolve(module, required):
        if not module or module.split('.')[0] not in LOCAL_PACKAGES:
            return
        parts = module.split('.')
        base = root.joinpath(*parts)
        source = base.with_suffix('.py')
        package = base / '__init__.py'
        if source.is_file():
            pending.append(source.relative_to(root).as_posix())
        elif package.is_file():
            pending.append(package.relative_to(root).as_posix())
        elif not base.is_dir() and required:
            missing.add(source.relative_to(root).as_posix())
        for end in range(1, len(parts)):
            init = root.joinpath(*parts[:end], '__init__.py')
            if init.is_file():
                pending.append(init.relative_to(root).as_posix())

    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = root / name
        if not path.is_file():
            missing.add(name)
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
        # Package initializers can expose unrelated lazy APIs. Inspect their
        # executed module-level statements, not unused function/class bodies.
        nodes = list(tree.body)
        while nodes:
            node = nodes.pop()
            if name.endswith('/__init__.py') and isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            nodes.extend(ast.iter_child_nodes(node))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolve(alias.name, True)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level:
                    package = '.'.join(Path(name).parts[:-1])
                    module = importlib.util.resolve_name('.' * node.level + module, package)
                resolve(module, True)
                for alias in node.names:
                    if alias.name != '*':
                        resolve(module + '.' + alias.name, False)
    if missing:
        raise RuntimeError('INCOMPLETE_SOURCE_DELIVERY: missing ' + ', '.join(sorted(missing)) +
                           '. Update the complete codex/top8-dual-scorer branch before retrying.')
    return sorted(seen)


def check_delivery(root=ROOT):
    return dependency_files(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        files = check_delivery(args.root)
    except (RuntimeError, SyntaxError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(dict(status='source_dependencies_complete', files=len(files),
                          model_imports=False, scientific_execution=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
