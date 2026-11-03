"""Per-agent factories and a single-writer lease, not an OS sandbox."""

import fcntl
import hashlib
import os
from pathlib import Path
from corecoder.agent import Agent
from corecoder.tools import ALL_TOOLS


def fresh_tools(*, allow_agent=False):
    return [type(template)() for template in ALL_TOOLS if allow_agent or template.name != "agent"]


def isolated_agent(model, *, allow_agent=False, **kwargs):
    return Agent(model, tools=fresh_tools(allow_agent=allow_agent), **kwargs)


class WorkspaceLease:
    def __init__(self, root, *, shared=False):
        directory = Path("/tmp") / ("researchcoder-workspace-leases-" + str(os.getuid()))
        directory.mkdir(mode=0o700, exist_ok=True)
        self.path = directory / hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()
        self.stream = None
        self.shared = shared

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            fcntl.flock(self.stream, (fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            self.stream = None
            raise RuntimeError("Workspace already leased by another task") from None
        return self

    def __exit__(self, *args):
        if self.stream:
            fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None
