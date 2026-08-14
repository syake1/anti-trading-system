"""Manual, file-based adapter for the stocknote analysis application."""

from .runner import ContractError, process_request

__all__ = ["ContractError", "process_request"]
