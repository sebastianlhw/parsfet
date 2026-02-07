## 2024-05-22 - Static Report CSP Constraints
**Vulnerability:** XSS risk in generated HTML reports via malicious input data.
**Learning:** The project generates standalone "No-Build" HTML files intended for local viewing. Traditional HTTP CSP headers are unavailable.
**Prevention:** Implemented CSP via `<meta>` tag.
- Removed `unsafe-inline` from `script-src` by generating a cryptographic nonce at runtime and injecting it into the template.
- Retained `unsafe-eval` in `script-src` to support the standard Alpine.js build (which uses `new Function()` for directives).
- Retained `unsafe-inline` in `style-src` (without nonce) to support `style` attributes required by Plotly and Alpine.js.
