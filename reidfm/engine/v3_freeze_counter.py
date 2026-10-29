"""V3-only persistent FastReID AdamWithFreezeLayer step counter.

The pinned FastReID stores its freeze counter in a closure rather than in
optimizer.state_dict(). Preserve that counter as an independent checkpointable.
This intentionally does not modify the pinned FastReID implementation.
"""
from __future__ import annotations

import inspect


def _counter_cell(optimizer):
    candidates = [getattr(type(optimizer), 'step', None),
                  getattr(optimizer, 'step', None)]
    visited = set()
    while candidates:
        func = candidates.pop()
        if func is None:
            continue
        func = getattr(func, '__func__', func)
        if not callable(func) or id(func) in visited:
            continue
        visited.add(id(func))
        wrapped = getattr(func, '__wrapped__', None)
        if wrapped is not None:
            candidates.append(wrapped)
        code = getattr(func, '__code__', None)
        cells = getattr(func, '__closure__', None)
        if code is None or cells is None:
            continue
        for name, cell in zip(code.co_freevars, cells):
            if name == 'cnt':
                return cell
            try:
                nested = cell.cell_contents
            except ValueError:
                continue
            if inspect.isfunction(nested) or inspect.ismethod(nested):
                candidates.append(nested)
    raise RuntimeError('Pinned AdamWithFreezeLayer counter cell not found')


class V3FreezeCounterCheckpointable:
    SCHEMA = 1

    def __init__(self, optimizer, freeze_iters):
        if type(freeze_iters) is not int or freeze_iters <= 0:
            raise ValueError('freeze_iters must be a positive integer')
        if type(optimizer).__name__ != 'AdamWithFreezeLayer':
            raise TypeError('Unsupported optimizer for V3 freeze counter')
        self._cell = _counter_cell(optimizer)
        self.freeze_iters = freeze_iters
        self.restored = False
        self._validate(self.count)

    def _validate(self, count):
        if type(count) is not int or not 0 <= count <= self.freeze_iters:
            raise ValueError('Invalid freeze counter value')

    @property
    def count(self):
        return self._cell.cell_contents

    def state_dict(self):
        count = self.count
        self._validate(count)
        return {'schema': self.SCHEMA,
                'freeze_iters': self.freeze_iters,
                'count': count}

    def load_state_dict(self, state):
        if type(state) is not dict or set(state) != {
            'schema', 'freeze_iters', 'count'
        }:
            raise ValueError('Invalid freeze checkpoint schema')
        if type(state['schema']) is not int or state['schema'] != self.SCHEMA:
            raise ValueError('Freeze checkpoint version mismatch')
        if type(state['freeze_iters']) is not int or state['freeze_iters'] != self.freeze_iters:
            raise ValueError('Freeze schedule mismatch')
        count = state['count']
        self._validate(count)
        self._cell.cell_contents = count
        if self.count != count:
            raise RuntimeError('Freeze counter restoration failed')
        self.restored = True
