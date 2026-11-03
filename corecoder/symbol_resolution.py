"""Bounded static definition lookup from an already observed Python reference."""

from __future__ import annotations

import ast
import keyword
import os
from pathlib import Path

from .locator import _Locator
from .observation_state import FILE_KINDS, current_observation_ids
from .source_search import _excluded
from .source_visibility import visible_end_line

SYMBOL_LIMITS = {"files": 8, "file_bytes": 200_000, "bytes": 1_000_000, "depth": 6, "ast_nodes": 40000, "observations": 32}
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return base + "." + node.attr if base else None
    return None


def _bindings(body, name):
    """Collect possible bindings in one lexical scope, without executing branches."""
    found, stack = [], [(node, False) for node in reversed(body)]
    while stack:
        node, conditional = stack.pop()
        if isinstance(node, _DEFINITIONS):
            if node.name == name:
                found.append((node, None, conditional))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                if bound in (name, "*"):
                    found.append((node, alias, conditional))
            continue
        if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            found.append((node, None, conditional))
        # Store/Delete bindings include assignments, loops, with-targets and comprehensions.
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            found.append((node, None, conditional))
        if isinstance(node, ast.ExceptHandler) and node.name == name:
            found.append((node, None, True))
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name:
            found.append((node, None, True))
        if isinstance(node, ast.MatchMapping) and node.rest == name:
            found.append((node, None, True))
        conditional = conditional or isinstance(node, (ast.If, ast.Try, ast.For, ast.AsyncFor, ast.While, ast.Match))
        stack.extend((child, conditional) for child in reversed(list(ast.iter_child_nodes(node))))
    return found


def _call_scopes(tree, symbol, start, end, at_line):
    calls, fallback, stack = [], [], [(tree, ())]
    while stack:
        node, scopes = stack.pop()
        line = getattr(node, "lineno", 0)
        visible = start <= line <= end and (at_line is None or line == at_line)
        if visible and isinstance(node, ast.Call) and _dotted(node.func) == symbol:
            calls.append((scopes, line))
        if visible and ((isinstance(node, (ast.Name, ast.Attribute)) and _dotted(node) == symbol)
                        or isinstance(node, _DEFINITIONS) and node.name == symbol.split(".")[0]
                        or isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == symbol.split(".")[0]):
            fallback.append((scopes, line))
        if isinstance(node, (*_DEFINITIONS, ast.Lambda)):
            body = [node.body] if isinstance(node, ast.Lambda) else node.body
            for child in ast.iter_child_nodes(node):
                stack.append((child, scopes + (node,) if child in body else scopes))
        elif isinstance(node, _COMPREHENSIONS):
            stack.extend((child, scopes + (node,)) for child in ast.iter_child_nodes(node))
        else:
            stack.extend((child, scopes) for child in ast.iter_child_nodes(node))
    return calls or fallback


class _Resolution:
    def __init__(self, state, evidence_ids, max_results):
        self.state, self.evidence_ids, self.max_results = state, evidence_ids, max_results
        self.limits = dict(SYMBOL_LIMITS)
        self.reader = _Locator(state.root, {}, {"max_file_bytes": self.limits["file_bytes"], "max_snippet_lines": 40,
                                               "max_snippet_chars": 4000, "context_lines": 3})
        self.modules, self.records, self.candidates, self.reasons = {}, [], [], []
        self.active = set()
        self.nodes = 0

    def reason(self, code):
        if code not in self.reasons:
            self.reasons.append(code)

    def safe(self, path):
        lexical = Path(os.path.abspath(path))
        if not lexical.is_relative_to(self.state.root) or _excluded(lexical.relative_to(self.state.root)):
            self.reason("excluded_scope")
            return False
        current = self.state.root
        for part in lexical.relative_to(self.state.root).parts:
            current = current / part
            if current.is_symlink():
                self.reason("symlink_excluded")
                return False
        if not lexical.resolve().is_relative_to(self.state.root):
            self.reason("outside_project")
            return False
        return True

    def read(self, path):
        path = Path(path)
        if not self.safe(path):
            return None
        if path in self.modules:
            return self.modules[path]
        if len(self.modules) >= self.limits["files"]:
            self.reason("file_limit")
            return None
        remaining = self.limits["bytes"] - self.reader.stats["bytes_read"]
        if remaining <= 1:
            self.reason("byte_limit")
            return None
        self.reader.limits["max_file_bytes"] = min(self.limits["file_bytes"], remaining - 1)
        data = self.reader.read(str(path))
        self.modules[path] = None
        if data["status"] != "read":
            self.reason(data["status"])
            self.emit(path, data, role="failed_read")
            return None
        if data["parse_status"] != "parsed":
            self.reason(data["parse_status"])
            self.emit(path, data, role="parse_failure")
            return None
        try:
            tree = ast.parse("\n".join(data["lines"]), filename=str(path))
            for _ in ast.walk(tree):
                self.nodes += 1
                if self.nodes > self.limits["ast_nodes"]:
                    self.reason("ast_node_limit")
                    return None
        except (SyntaxError, RecursionError):
            self.reason("parse_error")
            return None
        self.modules[path] = (data, tree)
        return data, tree

    def emit(self, path, data, node=None, role="definition", qualname=None):
        if len(self.records) >= self.limits["observations"]:
            self.reason("observation_limit")
            return None
        payload = {"status": data["status"], "requested_path": path.relative_to(self.state.root).as_posix(),
                   "resolved_path": str(path), "relative_path": path.relative_to(self.state.root).as_posix(),
                   "symbol_role": role}
        if data.get("file_sha256"):
            line = getattr(node, "lineno", 1)
            scope = None
            if isinstance(node, _DEFINITIONS):
                scope = {"name": node.name, "qualname": qualname or node.name,
                         "start_line": min([line, *(d.lineno for d in node.decorator_list)]),
                         "body_start_line": node.body[0].lineno, "end_line": node.end_lineno}
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope = next((function for function in data["functions"]
                                  if function["qualname"] == scope["qualname"]
                                  and function["body_start_line"] == scope["body_start_line"]), scope)
                payload.update({"symbol_name": qualname or node.name,
                                "definition_kind": "class" if isinstance(node, ast.ClassDef) else "function"})
            snippet = self.reader.snippet(data["lines"], line, scope)
            if scope:
                snippet["kind"] = "definition"
            payload.update({"file_sha256": data["file_sha256"], "parse_status": data["parse_status"],
                            "current_line_no": line, "function": scope if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
                            **snippet, "truncated": snippet["char_truncated"] or snippet["function_truncated"]})
        record = self.state.observe("file_excerpt", payload, self.evidence_ids)
        if record["id"] not in {item["id"] for item in self.records}:
            self.records.append(record)
        return record["id"]

    def module_parts(self, path):
        parts = list(path.relative_to(self.state.root).with_suffix("").parts)
        return parts[:-1] if path.name == "__init__.py" else parts

    def module(self, parts, names, chain, depth):
        if depth > self.limits["depth"]:
            self.reason("depth_limit")
            return
        base = self.state.root.joinpath(*parts)
        paths = [path for path in (base.with_suffix(".py"), base / "__init__.py") if self.safe(path) and path.is_file()]
        if len(paths) > 1:
            self.reason("module_package_collision")
        if not paths:
            if len(names) > 1:
                return self.module([*parts, names[0]], names[1:], chain, depth + 1)
            self.reason("module_not_in_project")
        for path in paths:
            loaded = self.read(path)
            if loaded is None:
                continue
            data, tree = loaded
            bindings = _bindings(tree.body, names[0])
            if not bindings and path.name == "__init__.py" and len(names) > 1:
                self.module([*parts, names[0]], names[1:], chain, depth + 1)
            else:
                self.resolve(path, data, bindings, names, chain, depth, ())

    def resolve(self, path, data, bindings, names, chain, depth, prefix):
        key = (str(path), tuple(prefix), tuple(names))
        if key in self.active:
            self.reason("import_cycle")
            return
        if depth > self.limits["depth"]:
            self.reason("depth_limit")
            return
        self.active.add(key)
        try:
            if not bindings:
                self.reason("binding_not_found")
            if len(bindings) > 1:
                self.reason("multiple_bindings")
            if len(bindings) > 64:
                self.reason("binding_limit")
            for node, alias, conditional in bindings[:64]:
                if len(self.candidates) >= self.max_results:
                    self.reason("result_limit")
                    break
                if conditional:
                    self.reason("conditional_binding")
                identifier = self.emit(path, data, node, "import_binding" if alias else "definition",
                                       ".".join([*prefix, getattr(node, "name", names[0])]))
                if identifier is None:
                    break
                route = [*chain, identifier]
                if isinstance(node, _DEFINITIONS):
                    if not names[1:]:
                        self.candidates.append({"definition_id": identifier, "chain_ids": route})
                    elif isinstance(node, ast.ClassDef):
                        self.resolve(path, data, _bindings(node.body, names[1]), names[1:], route, depth + 1, (*prefix, node.name))
                    else:
                        self.reason("dynamic_function_attribute")
                elif isinstance(node, ast.ImportFrom) and alias.name != "*":
                    package = self.module_parts(path)
                    if path.name != "__init__.py":
                        package = package[:-1]
                    if node.level:
                        if node.level > len(package):
                            self.reason("relative_import_outside_project")
                            continue
                        package = package[:len(package) - node.level + 1]
                    else:
                        package = []
                    parts = package + (node.module.split(".") if node.module else [])
                    if node.module is None and parts == self.module_parts(path) and names[1:]:
                        self.module([*parts, alias.name], names[1:], route, depth + 1)
                    else:
                        self.module(parts, [alias.name, *names[1:]], route, depth + 1)
                elif isinstance(node, ast.Import):
                    imported = alias.name.split(".")
                    consume = 1 if alias.asname else len(imported)
                    if not alias.asname and names[:consume] != imported or len(names) <= consume:
                        self.reason("module_object_or_dynamic_attribute")
                    else:
                        self.module(imported, names[consume:], route, depth + 1)
                else:
                    self.reason("wildcard_or_dynamic_binding")
        finally:
            self.active.remove(key)


def resolve_symbol(state, symbol, source_observation_id, evidence_ids, at_line=None, max_results=6):
    if (not isinstance(symbol, str) or not 1 <= len(symbol) <= 120
            or any(not part.isidentifier() or keyword.iskeyword(part) for part in symbol.split("."))):
        raise ValueError("symbol must be a Python name or dotted reference of at most 120 characters")
    if (not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 10
            or any(not isinstance(item, str) or item not in state.exposed_evidence_ids for item in evidence_ids)):
        raise ValueError("Provide 1 to 10 already exposed evidence IDs")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 8:
        raise ValueError("max_results must be an integer between 1 and 8")
    if not isinstance(source_observation_id, str) or source_observation_id not in current_observation_ids(state):
        raise ValueError("source_observation_id must name a current observed source excerpt")
    source = state.observations[source_observation_id]
    data = source["data"]
    span = data.get("snippet") or data
    end = visible_end_line(span)
    if (source["kind"] not in FILE_KINDS or not data.get("file_sha256") or not end
            or not str(data.get("resolved_path", "")).endswith(".py")):
        raise ValueError("Source observation must contain actual Python lines and a file hash")
    if at_line is not None and (isinstance(at_line, bool) or not isinstance(at_line, int)
                               or not span["start_line"] <= at_line <= end):
        raise ValueError("at_line must be inside the actually observed source range")
    resolver = _Resolution(state, evidence_ids, max_results)
    path = Path(data["resolved_path"])
    loaded = resolver.read(path)
    if loaded:
        actual, tree = loaded
        if actual["file_sha256"] != data["file_sha256"]:
            resolver.emit(path, actual, role="source_version_changed")
            resolver.reason("source_version_changed")
        else:
            refs = _call_scopes(tree, symbol, span["start_line"], end, at_line)
            scopes = {scope for scope, _ in refs}
            if not refs:
                resolver.reason("reference_not_in_observed_lines")
            elif len(scopes) > 1:
                resolver.reason("ambiguous_callsite_use_at_line")
            else:
                scopes = refs[0][0]
                bindings, prefix = [], ()
                for scope in reversed(scopes):
                    if isinstance(scope, _COMPREHENSIONS):
                        resolver.reason("comprehension_scope_unsupported")
                        break
                    if isinstance(scope, ast.ClassDef) and any(isinstance(s, _FUNCTIONS) for s in scopes):
                        continue
                    if isinstance(scope, _FUNCTIONS):
                        args = scope.args
                        parameters = [*args.posonlyargs, *args.args, *args.kwonlyargs, *([args.vararg] if args.vararg else []),
                                      *([args.kwarg] if args.kwarg else [])]
                        if symbol.split(".")[0] in {arg.arg for arg in parameters}:
                            resolver.reason("dynamic_parameter")
                            break
                    bindings = _bindings([scope.body] if isinstance(scope, ast.Lambda) else scope.body, symbol.split(".")[0])
                    if bindings:
                        prefix = tuple(getattr(s, "name", "<lambda>") for s in scopes[:scopes.index(scope) + 1])
                        break
                else:
                    bindings = _bindings(tree.body, symbol.split(".")[0])
                if not resolver.reasons:
                    resolver.resolve(path, actual, bindings, symbol.split("."), [], 0, prefix)
    candidates = list({item["definition_id"]: item for item in resolver.candidates}.values())
    summary = state.observe("symbol_resolution", {
        "status": "resolved" if len(candidates) == 1 and not resolver.reasons else "ambiguous" if candidates else "unresolved",
        "symbol": symbol, "source_observation_id": source_observation_id, "at_line": at_line,
        "candidates": candidates, "reasons": resolver.reasons, "limits": {**resolver.limits, "max_results": max_results},
        "stats": {"files": len(resolver.modules), "bytes_read": resolver.reader.stats["bytes_read"], "ast_nodes": resolver.nodes},
        "dependency_observation_ids": [item["id"] for item in resolver.records]
                                      + ([] if "source_version_changed" in resolver.reasons else [source_observation_id]),
        "scope": "Static lexical candidates only; branches, runtime dispatch, import hooks and sys.path changes are not evaluated. "
                 "No project module was imported or executed; missing candidates are scoped unknowns, not proof of absence.",
    }, evidence_ids)
    return {"observations": [*resolver.records, summary]}
