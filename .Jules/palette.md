## 2025-02-12 - [Alpine.js Keyboard Shortcuts]
**Learning:** Global shortcuts via `@keydown.window` are seamless but require manual filtering of input fields (`INPUT`, `TEXTAREA`). Tooltips on tabs (`title="Shortcut: 1"`) are a lightweight way to expose these.
**Action:** Reuse this pattern for panel switching and primary actions, always checking `e.target.tagName`.
