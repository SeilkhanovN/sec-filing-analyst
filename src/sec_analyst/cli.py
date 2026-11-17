"""Command-line entry point: python -m sec_analyst.cli <command>."""

from __future__ import annotations

import argparse
import json

from .config import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sec_analyst", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-config", help="show which settings are configured (never shows secrets)")
    args = parser.parse_args(argv)

    if args.command == "check-config":
        print(json.dumps(load_settings().describe(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
