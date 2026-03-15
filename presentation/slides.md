---
theme: apple-basic
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Vibe-Coding the Un-parsable
  A presentation on building Pars-FET.
drawings:
  persist: false
transition: auto
title: "Vibe-Coding the Un-parsable"
mdc: true
---

# Chaos to JSON
## Vibe-Coding the Un-parsable

A Case Study in Agentic Architecture: Building **Pars-FET**

---
transition: fade
layout: center
---

# The Problem: Apples, Oranges, and Nanoseconds

<v-click>

### The Blueprint Dilemma
You get blueprints from two different foundries:
*   **Foundry A**: Tells you transistor strength in "ox-pulls".
*   **Foundry B**: Tells you transistor strength in "Joules".

</v-click>

<v-click>

### The `.lib` Reality
The semiconductor industry runs on `.lib` files.
One says delay is `0.05`. Another says `50`. If we plot the raw numbers, we're comparing apples to space shuttles.

</v-click>

---
layout: default
---

# The Solution: The "Atom"

We stop caring about absolute "nanoseconds." We start caring about **ratios**.

<div class="mt-8 mb-6 max-w-2xl mx-auto border-l-4 border-blue-500 pl-6 bg-gray-50/50 p-6 rounded-r-lg shadow-sm">
  <h3 class="text-2xl font-light text-gray-800 tracking-wide">"How big are you compared to the Atom?"</h3>
</div>

<div class="grid grid-cols-2 gap-8 mt-12">
<div v-click>

### 1. Identify the Baseline
Find the Standard Inverter (e.g., `INVD1`). This is our "Atom."

### 2. Measure It
Set its size to **$1.0$**. Set its delay to **$1.0$**.

</div>
<div v-click>

### 3. Compare the Rest
If a NAND gate is twice as slow, its new universal metric is simply **`2.0`**.

### 4. The Universal Truth
A NAND gate is roughly twice as slow, whether it's 1980s tech or a 2030s quantum array.

</div>
</div>

---
layout: two-cols
---

# The Old Way
### *The "Eternal" Parsing Debt*

<br>

<v-clicks>

*   **The Medium**: `grep`, `sed`, and fragile Regex strings.
*   **The Experience**: "Extract A and B if it looks like X".
*   **The Cost**: A high cognitive tax. Every edge case broke the parser.
*   **The Code**: A messy black box that works, but is impossible to maintain.

</v-clicks>

::right::

# The Pars-FET Way
### *Data as a First-Class Citizen*

<br>

<v-clicks>

*   **The Medium**: Lexical Tokenization and an Abstract Syntax Tree (AST).
*   **The Experience**: "Understand the grammar to extract the metrics".
*   **The Cost**: High up-front architecture, zero maintenance debt.
*   **The Code**: Strictly typed **Python Dataclasses** ready for Plotting and DB ingestion.

</v-clicks>

---
transition: slide-up
layout: center
---

# The "Vibe" in Action
## How we built the Pars-FET Parser

<v-click>
    
### The Setup
*   **The Goal**: Translate a complex C-like nested file (`.lib`) into an AST.
*   **The Tool**: `Antigravity` (Agentic IDE).
*   **The "Vibe" Prompt**: *"Build a recursive descent parser. Identify groups versus attributes."*

</v-click>

---
layout: default
---

# "What I cannot create, I do not understand."
<div class="text-gray-400 text-sm mt-1 tracking-widest uppercase">- Richard Feynman</div>

<br>

To parse the text, `Antigravity` helped build a **Recursive Descent** parser.

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="text-left">

### 1. Tokenization
We chop `pin(A){` into `['pin', '(', 'A', ')', '{']`.

### 2. The Abstract Syntax Tree
Whenever the parser sees a `(`, it calls itself to build a branch on the tree.

</div>
<div>

```python {monaco-run}
# Live Interactive Parsing
from parsfet.parsers.liberty import LibertyParser

# We packaged Pars-FET and shipped it to WebAssembly!
parser = LibertyParser()

print("Reading and parsing /sample.lib from memory...")
with open('/sample.lib', 'r') as f:
    ast = parser.parse(f.read())

print("✅ Successfully parsed library:", ast["library"][0]["_qualifier"])
print("✅ Found", len(ast["library"][0]["cell"]), "cells.")
print("✅ First cell:", ast["library"][0]["cell"][0]["_qualifier"])
```

</div>
</div>

---
transition: fade
layout: center
---

# The Tool: Antigravity
### "With Great Power Comes... Memory Leaks?"

<br>

*   **Architecture**: High-performance fork of **VS Code**.
*   **Workspace Aware**: Maintains a holistic view; it sees the `parser_explanation.md` *and* the `.lib` specification.
*   **The Reality**: The model seeks the lowest "Token Cost."

<br>
<v-click>
    
> *"It prefers generating **new logic** over reading **existing infrastructure** to save context space."*
> <div class="text-right text-sm text-gray-400">- The "Lazy" Agent Observation</div>
    
</v-click>

---
layout: default
---

# ⚠️ When "Vibes" Go Toxic
### The Cost of Autonomy

<div class="grid grid-cols-2 gap-8 mt-10">

<div v-click>

### 1. The Full-File Hallucination
The AI makes a syntax error. To fix it, it predicts a rewrite of the entire file. The logic is subtly altered in the periphery.

</div>

<div v-click>

### 2. The "Infinite Revert" Dead-Loop
The agent enters a loop, gets stuck, and triggers an autonomous `git revert`. It realizes the last human commit was 4 hours ago.

</div>

</div>

<v-click>

<div class="mt-12 text-center text-red-500 font-bold text-2xl border border-red-200 p-4 rounded bg-red-50">
Result: 2 billion tokens of "progress" vanished in a millisecond.
</div>

</v-click>

---
layout: center
---

# 🛡️ Git is the Only Sane Recovery Strategy

<div class="text-xl font-light tracking-wide text-gray-600 mt-2 mb-10">
"Good practices don't slow you down; they allow you to safely go faster."
</div>

<v-clicks>

*   **Commit Small, Commit Frequent**: A "long time" safely coding is now 15 minutes.
*   **The Save-Point Strategy**: Commit *before* every major prompt to act as an experimental branch.
*   **Semantic History**: Distinguish human intent (`human: fix abstract logic`) vs. AI generation (`vibe: generate boilerplate`).

</v-clicks>

---
layout: center
class: text-center
transition: fade
---

# The Final Vibe

<br>
<br>

<div class="text-4xl font-light tracking-wide text-gray-800">
"The AI provides the <span class="font-semibold text-blue-600">Velocity</span>.
</div>

<div class="text-4xl font-light tracking-wide text-gray-800 mt-6">
You provide the <span class="font-semibold text-blue-600">Integrity</span>."
</div>
