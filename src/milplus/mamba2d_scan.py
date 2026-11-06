"""Explicit transformations of verified author 2D scan definitions; no vendored code."""
import ast
import types

import torch


def _clone(value):
    return value.clone() if torch.is_tensor(value) else value


def _safe_mul(left, right):
    return _clone(left)*_clone(right)


def _safe_add(left, right):
    return _clone(left)+_clone(right)


class FunctionalWrites(ast.NodeTransformer):
    def __init__(self):
        self.writes = 0
        self.products = 0

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in {'add_', 'mul_'}:
            if len(call.args) != 1 or call.keywords or not isinstance(call.func.value, ast.Subscript):
                raise ValueError('Pinned 2D scan mutation contract changed')
            target = ast.parse(ast.unparse(call.func.value), mode='eval').body
            target.ctx = ast.Store()
            value = ast.Call(func=ast.Name(id='_safe_add' if call.func.attr == 'add_' else '_safe_mul', ctx=ast.Load()),
                             args=[self.visit(call.func.value), self.visit(call.args[0])], keywords=[])
            self.writes += 1
            return ast.copy_location(ast.Assign(targets=[target], value=value), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'mul':
            if len(node.args) != 1 or node.keywords:
                raise ValueError('Pinned 2D scan product contract changed')
            self.products += 1
            return ast.copy_location(ast.Call(func=ast.Name(id='_safe_mul', ctx=ast.Load()),
                                              args=[node.func.value, *node.args], keywords=[]), node)
        return node


def build_scan(source_text):
    transform = FunctionalWrites()
    tree = ast.fix_missing_locations(transform.visit(ast.parse(source_text)))
    scope = {'_safe_mul': _safe_mul, '_safe_add': _safe_add}
    exec(compile(tree, 'verified_author_scan_explicit_derivative_repair.py', 'exec'), scope)  # noqa: S102 -- pinned Git definitions, documented mutation/derivative repair
    def scan(a, x):
        context = types.SimpleNamespace(save_for_backward=lambda *args: None)
        return scope['PScan_2D'].forward(context, a, x)
    return scan, {'rewritten_indexed_writes': transform.writes, 'cloned_products': transform.products}


def build_paper_scan(source_text):
    """Named forward correction to primary Eq5/Eq6, plus automatic derivatives."""
    changes = [
        ('Aa = A[:, :, 2**(num_steps-2)-1:H:2**(num_steps-2)]',
         'Aa = A_raw[:, :, 2**(num_steps-2)-1:H:2**(num_steps-2)]'),
        ('Aa = A[:, :, 2**k-1:H:2**k]', 'Aa = A_raw[:, :, 2**k-1:H:2**k]'),
    ]
    for before, after in changes:
        if source_text.count(before) != 1:
            raise ValueError('Pinned 2D vertical downsweep contract changed')
        source_text = source_text.replace(before, after)
    scan, proof = build_scan(source_text)
    return scan, {**proof, 'forward_repairs': 'vertical downsweep uses the original transition map',
                  'equations': 'arXiv2412.00678v3 Eq5 and Eq6; independently validated'}
