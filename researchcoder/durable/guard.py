"""Parent-death process-group cleanup. This is not an OS security sandbox."""

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parent = int(sys.argv[1])
    command = sys.argv[2:]

    def terminate(*_):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGUSR1, terminate)
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGUSR1, 0, 0, 0) != 0:
        return 2
    if os.getppid() != parent:
        terminate()
    fence = Path(command[-2]).parent / "lease.ready"
    deadline = time.monotonic() + 10
    while not fence.exists():
        if time.monotonic() > deadline:
            return 2
        time.sleep(0.005)
    child = subprocess.Popen(command, close_fds=True)
    return child.wait()


if __name__ == "__main__":
    sys.exit(main())
