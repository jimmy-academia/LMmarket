
## 🧭 Coding Philosophies (for fast, reliable research code)

**Goal:** ship small, durable utilities that make runs reproducible, debuggable, and low-friction.

### 1) Keep helpers tiny

* Prefer single-purpose utilities **≤10 lines**.
* If a helper needs more than ~10–15 lines, split it or raise the abstraction.
* Favor the **standard library** (`pathlib`, `logging`, `argparse`, `json`) over frameworks.

### 2) Clear naming by intent

* Side-effects: `set_*`, `ensure_*`, `write_*`.
* Values: nouns (`dset_root`, `output_dir`).
* “File that stores a value”: add `*_ref` (e.g., `dset_root_ref`, `openai_key_ref`).
* Paths vs dirs: use `*_path` (file) and `*_dir` (directory).

### 3) Defaults that just work

* All ephemeral outputs under `cache/` (gitignored): `cache/logs`, `cache/meta`, etc.
* Use `Path(...).mkdir(parents=True, exist_ok=True)` for idempotent setup.
* Global `logging` (no per-module loggers) with a single `set_verbose(verbose)`.

### 4) Logging over print

* One call at startup configures both **console + file** under `cache/logs/`.
* Map `verbose` to `WARNING/INFO/DEBUG`.
* Messages are short and action-oriented.

### 5) Human-in-the-loop when needed

* If a required value is missing, **warn once**, prompt succinctly, save it, continue.
* In **non-interactive** contexts, **fail fast with a copy-pasteable fix** in the error.
* Never loop endlessly; one prompt, then proceed or exit.

### 6) Reproducibility first

* `set_seeds(seed)` sets Python/NumPy/PyTorch and cudnn flags.
* Optionally log the git commit and the resolved paths at startup.
* Return useful artifacts (e.g., `log_path`) only if the caller might record them.

### 7) Minimal signatures, predictable types

* Accept `str | Path`; return `Path` (or plain `str`) consistently within a module.
* Avoid passing logger objects; rely on `import logging` after a single setup call.

### 8) Errors that teach

* Errors are **one line of cause** + **one line of remedy** (e.g., shell commands).
* Warnings precede interactive prompts; infos confirm saved state.

### 9) Avoid configuration sprawl

* Prefer simple CLI flags + tiny helpers over config frameworks.
* Keep docstrings and comments focused on **why**, not how.

### 10) Performance last, clarity first

* Don’t micro-optimize helpers; measure before tuning.
* Localize heavy imports (inside functions) only when they materially reduce startup time.

---

### Quick patterns (copyable)

```python
# Ensure directory (idempotent, returns Path)
from pathlib import Path
def ensure_dir(p): p = Path(p); p.mkdir(parents=True, exist_ok=True); return p
```

```python
# Verbosity → logging (console + file)
import os, logging
from datetime import datetime
def set_verbose(verbose=1, log_dir="cache/logs", prefix="exp"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S"); lp=f"{log_dir}/{prefix}_{ts}.log"
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    lvl = logging.WARNING if verbose<=0 else (logging.INFO if verbose==1 else logging.DEBUG)
    logging.basicConfig(level=lvl, format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(),
                                                      logging.FileHandler(lp,"w","utf-8")])
    logging.info(f"Logging → {lp}")
```

```python
# Pathref (file that stores a value) — read if exists, else prompt once and save
from pathlib import Path; import logging
def ensure_pathref(ref):
    ref = Path(ref)
    if ref.is_file(): return ref.read_text().strip()
    logging.warning(f"[ensure_pathref] {ref} not found"); val = input(f"...enter content for {ref}: ").strip()
    ref.parent.mkdir(parents=True, exist_ok=True); ref.write_text(val); logging.info(f"[ensure_pathref] saved → {ref}")
    return val
```

---

**Principle in one line:**

> Small, single-purpose helpers + plain stdlib + clear names + idempotent I/O + human-friendly failure paths.
