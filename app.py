"""Launch the GUI or its isolated worker from one frozen executable."""

from __future__ import annotations

import sys


def _restore_worker_output() -> None:
    """Reconnect stdout/stderr when a windowed PyInstaller EXE is a worker."""

    if sys.platform != "win32":
        return
    import ctypes
    import os
    import msvcrt

    for name, handle_number in (("stdout", -11), ("stderr", -12)):
        if getattr(sys, name, None) is not None:
            continue
        handle = ctypes.windll.kernel32.GetStdHandle(handle_number)
        if not handle or handle == ctypes.c_void_p(-1).value:
            continue
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY)
            stream = open(
                descriptor,
                "w",
                encoding="utf-8",
                errors="replace",
                buffering=1,
                closefd=False,
            )
        except OSError:
            continue
        setattr(sys, name, stream)


def main() -> int:
    if "--worker" in sys.argv[1:]:
        _restore_worker_output()
        arguments = list(sys.argv[1:])
        arguments.remove("--worker")
        from ai_photogrammetry.engineering.worker import main as worker_main

        return worker_main(arguments)

    smoke_test = "--smoke-test" in sys.argv[1:]
    if smoke_test:
        _restore_worker_output()
        print("startup: importing desktop", flush=True)

    from ai_photogrammetry.engineering.desktop import main as desktop_main

    if smoke_test:
        print("startup: desktop imported", flush=True)

    return desktop_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
