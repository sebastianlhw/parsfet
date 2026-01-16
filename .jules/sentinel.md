# Sentinel's Journal

## 2025-02-14 - Denial of Service in File Format Detection
**Vulnerability:** The `detect_format` function in `src/parsfet/cli.py` was reading the entire file into memory using `path.read_text()` just to check the first 1000 characters.
**Learning:** Functions that detect file formats or metadata should be extremely careful about resource usage. Even for a seemingly harmless check, assume the input file could be malicious (e.g., infinite stream or massive file).
**Prevention:** Use streaming or partial reads (e.g., `f.read(1000)`) instead of loading whole files. Always consider the worst-case input size.
