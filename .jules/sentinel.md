## 2025-02-19 - [JSON Injection in HTML Templates]
**Vulnerability:** XSS vulnerability found in `html_generator.py` where JSON data was directly injected into a `<script>` tag using string replacement.
**Learning:** `json.dumps()` alone is insufficient for embedding JSON in HTML `<script>` tags because it does not escape `/`, allowing `</script>` to terminate the block.
**Prevention:** Always escape `<` characters (e.g., replace with `\u003c`) when embedding JSON into HTML.
