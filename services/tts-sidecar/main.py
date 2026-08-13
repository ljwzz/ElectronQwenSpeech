from __future__ import annotations

import sys
from contextlib import redirect_stdout

from tts_sidecar.service import SidecarService


def main() -> int:
    protocol_stdout = sys.stdout
    with redirect_stdout(sys.stderr):
        service = SidecarService(stdout=protocol_stdout, stderr=sys.stderr)
        shutdown_requested = False
        try:
            for raw_line in sys.stdin:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                shutdown_requested = service.dispatch_line(line)
                if shutdown_requested:
                    break
        finally:
            if not shutdown_requested:
                service.close_input()
            service.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
