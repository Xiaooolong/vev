"""Command-line entry point: `vev serve ...`."""

from __future__ import annotations

import sys

USAGE = """usage: vev <command> [options]

commands:
  serve      run the /v1/systemone server (vev serve --help)
  version    print the installed version
"""


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(USAGE, end="")
        return
    cmd, rest = args[0], args[1:]
    if cmd == "serve":
        from vev.serve import main as serve_main

        serve_main(rest)
    elif cmd == "version":
        from vev import __version__

        print(__version__)
    else:
        sys.exit(f"vev: unknown command {cmd!r}\n{USAGE}")


if __name__ == "__main__":
    main()
