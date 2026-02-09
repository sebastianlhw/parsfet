
import logging
from parsfet.log_utils import setup_logging
from rich.logging import RichHandler

def test_log_markup_disabled():
    # Reset logging handlers
    root = logging.getLogger()
    root.handlers = []

    setup_logging()

    handler = root.handlers[0]
    assert isinstance(handler, RichHandler)
    # Accessing the markup property of RichHandler
    # RichHandler stores it in self.markup if passed to __init__?
    # Let's check Rich source or just assume it sets it.
    # RichHandler definition: def __init__(..., markup: bool = False, ...)
    # It sets self.markup = markup

    assert handler.markup is False
