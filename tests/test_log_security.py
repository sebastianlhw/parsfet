import logging
from parsfet.log_utils import setup_logging
from rich.logging import RichHandler

def test_log_injection_mitigation(capsys):
    """Test that log injection via Rich markup is mitigated."""
    # Reset logging handlers
    root_logger = logging.getLogger()
    # print(f"Initial handlers: {root_logger.handlers}")
    root_logger.handlers = []

    try:
        # Setup logging (this should add RichHandler with markup=False after fix)
        setup_logging()

        # Verify handler was added
        # print(f"Handlers after setup: {root_logger.handlers}")
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], RichHandler)
        assert root_logger.handlers[0].markup is False

        # Force the level to INFO just in case
        root_logger.setLevel(logging.INFO)

        # Log a message with markup
        markup_message = "[bold red]MALICIOUS[/bold red]"
        logging.info(markup_message)

        # Capture output
        captured = capsys.readouterr()
        output = captured.err

        # If captured.err is empty, check stdout just in case
        if not output:
             output = captured.out

        # print(f"Captured output: {output}")

        assert "[bold red]" in output
        assert "[/bold red]" in output
        assert "MALICIOUS" in output

    finally:
        pass
