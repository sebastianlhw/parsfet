"""Logging utilities for Pars-FET."""

import logging
from rich.logging import RichHandler

def setup_logging(quiet: bool = False) -> None:
    """Configures the logging for the application.

    Args:
        quiet: If True, set log level to WARNING (show only warnings/errors).
               If False (default), set to DEBUG (verbose mode).
    """
    # Default to DEBUG (verbose), unless --quiet is specified
    level = logging.WARNING if quiet else logging.DEBUG
    
    # Only configure if not already configured (prevents multiple calls)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)]
        )
    else:
        # Update level if already configured
        logging.getLogger().setLevel(level)
