## 2026-02-09 - Alpine.js CSP and Inline Expressions
**Learning:** The project was using `@alpinejs/csp` build but included inline expressions (e.g., `x-show="activeTab === 'browser'"`). This combination prevents Alpine from evaluating expressions, breaking interactivity completely without throwing obvious errors until runtime.
**Action:** When using Alpine with inline expressions in a no-build HTML template, use the standard build and ensure `'unsafe-eval'` is allowed in CSP, or refactor to move all logic into `Alpine.data()`.
