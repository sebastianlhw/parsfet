# Sentinel's Journal

## 2025-02-12 - Rich Logging Markup Injection
**Vulnerability:** The logging handler `RichHandler` was configured with `markup=True`, allowing attackers to inject formatting codes (colors, bold) into log files via user-controlled input (e.g., file names). This enables log spoofing/integrity attacks.
**Learning:** Python logging libraries like `rich` prioritize developer experience (colors) over security by default or through common copy-paste configurations. Mixing data and formatting instructions in the same channel is inherently risky.
**Prevention:** Explicitly configure `RichHandler(markup=False)` in `log_utils.py` and enforce regression testing (`tests/test_log_security.py`) to ensure this setting is never reverted.
