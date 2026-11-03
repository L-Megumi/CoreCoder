"""Bounded static initialization leads; never evaluate defaults or execute source."""

from __future__ import annotations

import ast

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _own_nodes(function):
    stack = list(reversed(function.body))
    visited, nodes = 0, []
    while stack and visited < 4000:
        node = stack.pop()
        visited += 1
        if isinstance(node, (*_FUNCTIONS, ast.ClassDef, ast.Lambda)):
            continue
        nodes.append(node)
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return nodes, bool(stack)


def initialization_lead(owner, method, qualname):
    """Only directly defined instance methods and literal receiver attributes."""
    if method.name == "__init__" or method.decorator_list:
        return None
    receiver = method.args.posonlyargs + method.args.args
    if not receiver:
        return None
    receiver = receiver[0].arg
    method_nodes, method_truncated = _own_nodes(method)
    attributes = sorted({node.attr for node in method_nodes
                         if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                         and isinstance(node.value, ast.Name) and node.value.id == receiver})
    if not attributes:
        return None
    result = {"class_name": owner.name, "method": qualname, "attributes": attributes[:12],
              "attributes_omitted": max(0, len(attributes) - 12),
              "scope": "Static source leads only; initialization execution and interface intent are not proved"}
    if method_truncated:
        return {**result, "status": "unknown", "reason": "Method AST analysis bound reached"}
    constructors = [node for node in owner.body if isinstance(node, _FUNCTIONS) and node.name == "__init__"]
    if len(constructors) != 1 or constructors[0].decorator_list:
        return {**result, "status": "unknown", "reason": "No unique undecorated local initializer; inheritance/dynamic binding unknown"}
    constructor = constructors[0]
    args = constructor.args.posonlyargs + constructor.args.args
    if not args:
        return {**result, "status": "unknown", "reason": "Initializer receiver unavailable"}
    init_receiver = args[0].arg
    assignments = {}
    constructor_nodes, constructor_truncated = _own_nodes(constructor)
    if constructor_truncated:
        return {**result, "status": "unknown", "reason": "Initializer AST analysis bound reached"}
    for node in constructor_nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for attribute in ast.walk(target):
                    if (isinstance(attribute, ast.Attribute) and isinstance(attribute.ctx, ast.Store)
                            and isinstance(attribute.value, ast.Name) and attribute.value.id == init_receiver
                            and attribute.attr in attributes):
                        assignments.setdefault(attribute.attr, []).append((node.lineno, node.end_lineno))
    if not assignments:
        return {**result, "status": "unknown", "reason": "Observed attributes lack direct local initializer assignments"}
    header_end = max(constructor.lineno, constructor.body[0].lineno - 1)
    intervals = [(constructor.lineno, min(header_end, constructor.lineno + 19))]
    intervals.extend(interval for spans in assignments.values() for interval in spans)
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    total = sum(end - start + 1 for start, end in merged)
    required = []
    for start, end in merged:
        required.extend(range(start, min(end + 1, start + 32 - len(required))))
        if len(required) == 32:
            break
    return {**result, "status": "candidate", "initializer": qualname.rsplit(".", 1)[0] + ".__init__",
            "required_lines": required, "required_lines_omitted": max(0, total - 32),
            "header_truncated": header_end > constructor.lineno + 19,
            "unresolved_attributes": [name for name in attributes if name not in assignments][:12]}


def initialization_source_lines(functions, lines, start, end):
    required = {number for function in functions for number in (function.get("initialization_check") or {}).get("required_lines", [])}
    selected = sorted(number for number in required if start <= number <= end)
    return [{"line_no": number, "text": lines[number - 1]} for number in selected[:32]]
