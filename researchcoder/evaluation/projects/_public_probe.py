"""Standalone, stdlib-only observation of a protected public Python checker.

This file is executed by the checker's interpreter, which need not have
ResearchCoder installed. It is not an OS sandbox. The controller independently
checks its exclusive receipt and never trusts text printed by candidate code.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import stat
import sys
import sysconfig
import traceback
import types


MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 20_000_000
MAX_FILES = 4000
MAX_ENTRIES = 20000
MAX_FRAMES = 128
MAX_DEPENDENCY_ENTRIES = 200_000


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class _RecordedStream:
    """Observe ordinary Python text output without redirecting or hiding it.

    Unobserved native/binary writes cause receipt validation to fail closed;
    this object does not claim to intercept arbitrary file-descriptor writes.
    """
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, text):
        count = self.stream.write(text)
        if count is not None and count != len(text):
            raise ValueError("Partial public output write")
        self.digest.update(text.encode("utf-8"))
        return count

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _regular(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Probe refuses symlinks")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
        raise ValueError("Probe refuses unsupported files")
    raw = path.read_bytes()
    after = path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(raw) != before.st_size):
        raise ValueError("Probe source changed during capture")
    return raw


def _capture(root):
    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Probe candidate root is unavailable")
    records, source = [], {}
    count = size = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as stream:
            for item in stream:
                count += 1
                if count > MAX_ENTRIES or item.is_symlink():
                    raise ValueError("Probe candidate entry bound or symlink")
                path = Path(item.path)
                if item.is_dir(follow_symlinks=False):
                    stack.append(path)
                    continue
                if not item.is_file(follow_symlinks=False) or len(records) >= MAX_FILES:
                    raise ValueError("Probe candidate file bound or unsupported type")
                raw = _regular(path)
                size += len(raw)
                if size > MAX_TOTAL_BYTES:
                    raise ValueError("Probe candidate byte bound")
                relative = path.relative_to(root).as_posix()
                records.append(dict(path=relative, sha256=_sha(raw), size_bytes=len(raw), schema_version=1))
                if path.suffix == ".py":
                    source[str(path)] = raw
    if not records:
        raise ValueError("Probe candidate is empty")
    records.sort(key=lambda r: r["path"])
    return records, source


def _code_sha(code):
    # marshal's interned-string reference flags can differ between an imported
    # function and a freshly compiled one. Normalize semantic code fields so a
    # real frame is compared to source, not only its forgeable co_filename.
    def constant(value):
        if isinstance(value, types.CodeType):
            return ["code", fields(value)]
        if isinstance(value, tuple):
            return ["tuple", [constant(item) for item in value]]
        if isinstance(value, frozenset):
            return ["frozenset", sorted((constant(item) for item in value), key=_json)]
        if isinstance(value, bytes):
            return ["bytes", value.hex()]
        if value is Ellipsis:
            return ["ellipsis"]
        if value is None or type(value) in (bool, str, int, float, complex):
            return [type(value).__name__, repr(value)]
        raise ValueError("Unsupported compiled constant")

    def fields(value):
        result = {name: getattr(value, name, None) for name in (
            "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize",
            "co_flags", "co_filename", "co_name", "co_qualname", "co_firstlineno")}
        for name in ("co_code", "co_linetable", "co_lnotab", "co_exceptiontable"):
            result[name] = getattr(value, name, b"").hex()
        for name in ("co_names", "co_varnames", "co_freevars", "co_cellvars"):
            result[name] = list(getattr(value, name))
        result["co_consts"] = [constant(item) for item in value.co_consts]
        return result
    return _sha(_json(fields(code)).encode("utf-8"))


def _compiled_codes(raw, filename):
    stack = [compile(raw, filename, "exec", dont_inherit=True, optimize=sys.flags.optimize)]
    result = set()
    while stack:
        code = stack.pop()
        result.add(_code_sha(code))
        stack.extend(value for value in code.co_consts if isinstance(value, types.CodeType))
    return result


def _stat_identity(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def _dependency_snapshot(root):
    """Metadata-only inventory of preexisting interpreter package sources.

    Roots come from this interpreter before candidate execution. Symlinks and
    incomplete inventories cannot authorize dependency-boundary attribution.
    Only traceback-relevant files are subsequently read and source-code matched.
    """
    paths = sysconfig.get_paths()
    roots = sorted({str(Path(paths[name]).resolve()) for name in ("purelib", "platlib") if paths.get(name)})
    inventory = {}
    count = 0
    complete = True
    try:
        for name in roots:
            package_root = Path(name)
            if (not package_root.is_dir() or package_root.is_relative_to(root)
                    or root.is_relative_to(package_root)
                    or any(p.is_symlink() for p in (package_root, *package_root.parents))):
                complete = False
                continue
            stack = [package_root]
            while stack:
                directory = stack.pop()
                with os.scandir(directory) as stream:
                    for item in stream:
                        count += 1
                        if count > MAX_DEPENDENCY_ENTRIES:
                            raise ValueError("Dependency inventory bound")
                        if item.is_symlink():
                            continue
                        path = Path(item.path)
                        if item.is_dir(follow_symlinks=False):
                            stack.append(path)
                        elif item.is_file(follow_symlinks=False) and path.suffix == ".py":
                            state = _stat_identity(path)
                            if state[2] <= MAX_FILE_BYTES:
                                inventory[str(path)] = dict(root=name, state=state)
    except (OSError, ValueError):
        complete = False
    public = dict(roots=roots, complete=complete, entries=len(inventory),
                  inventory_sha256=_sha(_json(inventory).encode("utf-8")),
                  interpreter_executable=str(Path(sys.executable).resolve()),
                  interpreter_prefix=str(Path(sys.prefix).resolve()))
    return inventory if complete else {}, public


def _frames(tb, root, source, dependencies, protected_paths):
    frames = []
    while tb is not None:
        if len(frames) >= MAX_FRAMES:
            raise ValueError("Probe traceback frame bound")
        code = tb.tb_frame.f_code
        filename = code.co_filename
        path = Path(filename)
        candidate_path = None
        source_sha = None
        code_sha = None
        matched = False
        dependency_path = dependency_root = dependency_before = dependency_after = None
        if path.is_absolute() and path.is_relative_to(root):
            candidate_path = path.relative_to(root).as_posix()
            raw = source.get(str(path))
            if raw is not None:
                source_sha = _sha(raw)
                code_sha = _code_sha(code)
                matched = code_sha in _compiled_codes(raw, str(path))
        elif filename in dependencies and filename not in protected_paths:
            bound = dependencies[filename]
            dependency_path, dependency_root = filename, bound["root"]
            dependency_before = bound["state"]
            dependency_after = _stat_identity(path)
            if dependency_before == dependency_after:
                raw = _regular(path)
                source_sha = _sha(raw)
                code_sha = _code_sha(code)
                matched = code_sha in _compiled_codes(raw, str(path))
        frames.append(dict(filename=filename, line=tb.tb_lineno, function=code.co_name,
                           candidate_path=candidate_path, source_sha256=source_sha,
                           code_sha256=code_sha, source_code_matched=matched,
                           dependency_path=dependency_path, dependency_root=dependency_root,
                           dependency_before=dependency_before, dependency_after=dependency_after))
        tb = tb.tb_next
    return frames


def execute(request_path, checker_argument):
    request_path = Path(request_path).resolve(strict=True)
    request_bytes = _regular(request_path)
    request = json.loads(request_bytes)
    if request.get("schema_version") != 1 or request.get("mode") != "candidate-exception-v1":
        raise ValueError("Unsupported public execution request")
    root = Path(request["candidate_root"])
    checker = Path(request["checker_path"])
    if checker_argument != str(checker):
        raise ValueError("Public checker argument does not match protected request")
    for protected in request["protected_files"]:
        if _sha(_regular(protected["path"])) != protected["sha256"]:
            raise ValueError("Protected public checker changed")
    before, source = _capture(root)
    dependencies, dependency_snapshot = _dependency_snapshot(root)
    receipt_path = request_path.parent / "execution-receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("Public execution receipt already exists")
    base = dict(schema_version=1, kind="public-python-execution-v1", mode=request["mode"],
                invocation_id=request["invocation_id"], job_id=request["job_id"], task_id=request["task_id"],
                request_sha256=_sha(request_bytes), candidate_root=str(root),
                original_argv=request["original_argv"], checker_path=str(checker),
                checker_sha256=_sha(_regular(checker)), candidate_before=before,
                dependency_snapshot=dependency_snapshot)
    # The checker's own argv and __main__ semantics stay intact. The trusted
    # interpreter remains isolated (-I); no new import search path is granted.
    sys.argv = [str(checker), *request["checker_arguments"]]
    outcome = "completed"
    exception_type = exception_module = None
    frames = []
    stdout, stderr = _RecordedStream(sys.stdout), _RecordedStream(sys.stderr)
    sys.stdout, sys.stderr = stdout, stderr
    try:
        runpy.run_path(str(checker), run_name="__main__")
    except BaseException as exc:
        outcome = "system_exit" if isinstance(exc, SystemExit) else "exception"
        exception_type, exception_module = type(exc).__name__, type(exc).__module__
        frames = _frames(exc.__traceback__, root, source, dependencies,
                         {p["path"] for p in request["protected_files"]})
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            raise
        # Render the actual traceback once, then exit one. Letting CPython render
        # it after finally would leave the receipt unable to bind the raw stderr
        # to this invocation (and could accidentally accept a stale sidecar).
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=stderr)
        raise SystemExit(1) from None
    finally:
        sys.stdout, sys.stderr = stdout.stream, stderr.stream
        after, _ = _capture(root)
        for protected in request["protected_files"]:
            if _sha(_regular(protected["path"])) != protected["sha256"]:
                raise ValueError("Protected public checker changed during execution")
        value = dict(base, candidate_after=after, outcome=outcome, exception_type=exception_type,
                     exception_module=exception_module, frames=frames,
                     stdout_sha256=stdout.digest.hexdigest(), stderr_sha256=stderr.digest.hexdigest())
        data = _json(value).encode("utf-8")
        if len(data) > 2_000_000:
            raise ValueError("Public execution receipt exceeds bound")
        with receipt_path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Expected protected execution request and public checker")
    execute(sys.argv[1], sys.argv[2])
