"""Immutable, source-bound training journals. No scientific import on inspection."""
from __future__ import annotations
import json
from pathlib import Path
from applications.rag.qvalue_portable_v001 import need, digest, checked_member, json_file, tensor_file


def source_binding():
    """Bind all shipped research modules, including dynamically imported controls."""
    root = Path(__file__).resolve().parent
    return {p.name: digest(p) for p in sorted(root.glob('q*.py')) +
            sorted(root.glob('top8_*.py'))}


class Journal:
    def __init__(self, root, binding):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root/'binding.json'
        if p.exists():
            need(json.loads(p.read_bytes()) == binding, 'resume configuration/source/cache mismatch')
        else:
            need(not any(self.root.iterdir()), 'unbound journal is not empty')
            json_file(self.root, 'binding.json', binding)
        self.binding = binding

    def latest_record(self):
        records = sorted(self.root.glob('step-*.json'))
        if not records:
            return None
        p = records[-1]
        record = json.loads(p.read_bytes())
        need(record['binding'] == self.binding and record['step'] > 0,
             'bound positive recovery step')
        checked_member(self.root, record['tensor'])
        return record

    def load(self):
        import torch
        record = self.latest_record()
        if record is None:
            return None
        value = torch.load(checked_member(self.root, record['tensor']),
                           map_location='cpu', weights_only=True)
        need(value['step'] == record['step'], 'recovery step identity')
        return value

    def save(self, value):
        step = value['step']
        name = f'step-{step:08d}'
        # Interrupted unsealed bytes remain intact; new generations never overwrite.
        generation = 0
        while (self.root/f'{name}-g{generation}.bin').exists() or \
              (self.root/f'{name}-g{generation}.bin.part').exists():
            generation += 1
        tensor = tensor_file(self.root, f'{name}-g{generation}.bin', value)
        json_file(self.root, name+'.json',
                  dict(step=step, binding=self.binding, tensor=tensor))


def new_generation(cell):
    cell = Path(cell)
    for generation in range(10000):
        p = cell/f'run-{generation:04d}'
        if not p.exists():
            p.mkdir()
            return p
    raise RuntimeError('recovery generation limit')
