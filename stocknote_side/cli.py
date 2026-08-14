"""Manual CLI for a single Phase 2 stocknote request file."""
from __future__ import annotations

import argparse
import importlib

from .runner import ContractError, process_request


def _load(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("analyzer must use module:function syntax")
    return getattr(importlib.import_module(module_name), function_name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Process one stocknote_request_<run_id>.json file")
    parser.add_argument("request", help="exactly one request JSON file")
    parser.add_argument("--analyzer", default="stocknote_side.analysis:analyze_candidate",
                        help="existing stocknote analyzer as module:function")
    parser.add_argument("--force", action="store_true", help="replace an existing response")
    args = parser.parse_args(argv)
    try:
        output = process_request(args.request, _load(args.analyzer), force=args.force)
    except (ContractError, FileExistsError, ImportError, AttributeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
