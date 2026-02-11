## 2026-02-11 - Rich Log Injection
**Vulnerability:** Log Injection / Terminal Formatting Injection via `RichHandler` in `log_utils.py`.
**Learning:** `RichHandler` defaults to `markup=True` (or was explicitly set to `True` here), which allows user input (like filenames) logged via standard `logging` calls to be interpreted as Rich markup. This enables attackers to inject formatting codes (color, bold) or even links into log output, potentially spoofing log entries or misleading operators.
**Prevention:** Always set `markup=False` when configuring `RichHandler` for general-purpose logging where untrusted input might be included in log messages. Use `rich.markup.escape()` if markup is absolutely necessary for specific parts of the message.
