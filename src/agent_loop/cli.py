"""Command-line entry point. Subcommands are wired up as the components land."""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_subparsers(dest="command")
    parser.parse_args(argv)
    parser.print_help()
    return 0
