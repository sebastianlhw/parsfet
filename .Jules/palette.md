# Palette's UX Journal 🎨

## 2024-05-23 - Accessibility in Data Visualization Dashboards
**Learning:** Interactive elements like `<summary>` and `<select>` often lack keyboard focus indicators by default, making them inaccessible to keyboard users. Using `focus-visible` pseudo-class allows for custom focus rings without annoying mouse users.
**Action:** Always verify `focus-visible` styles on interactive elements, especially those that trigger expanding/collapsing content or filtering data.
