"""Security tests for logging configuration."""

import logging
from rich.logging import RichHandler
from parsfet.log_utils import setup_logging

def test_log_markup_disabled():
    """Verify that RichHandler is configured with markup=False to prevent log injection."""

    # 1. Reset logging configuration to ensure we test a fresh setup
    root_logger = logging.getLogger()
    # Remove all existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # 2. Call the setup function
    setup_logging()

    # 3. Verify the handler configuration
    handlers = root_logger.handlers
    assert len(handlers) > 0, "Logging should have at least one handler configured"

    rich_handler = next((h for h in handlers if isinstance(h, RichHandler)), None)
    assert rich_handler is not None, "RichHandler should be configured"

    # CRITICAL SECURITY CHECK: Markup must be disabled
    assert rich_handler.markup is False, (
        "RichHandler.markup must be False to prevent log injection vulnerabilities. "
        "Do not enable markup for user-controlled log messages."
    )
