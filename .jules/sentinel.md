## 2024-05-22 - Static Report CSP Constraints
**Vulnerability:** XSS risk in generated HTML reports via malicious input data.
**Learning:** The project generates standalone "No-Build" HTML files intended for local viewing. Traditional HTTP CSP headers are unavailable.
**Prevention:** Implemented CSP via `<meta>` tag. Due to the architecture (single-file distribution with embedded JSON data and Alpine.js), `unsafe-inline` is required for scripts and styles, preventing full XSS mitigation but successfully restricting external resource loading to known CDNs.
