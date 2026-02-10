## 2025-02-10 - Subresource Integrity (SRI) for CDN Scripts
**Vulnerability:** External scripts loaded from CDNs without integrity checks can be compromised to inject malicious code (XSS/Supply Chain Attack).
**Learning:** The project relies on "No-Build" HTML reports using CDNs (Tailwind, Plotly, Alpine). This makes it vulnerable to CDN compromise.
**Prevention:** Always include `integrity="sha384-[HASH]"` and `crossorigin="anonymous"` attributes for all external script tags. Verified hashes using `curl | openssl` are critical.
