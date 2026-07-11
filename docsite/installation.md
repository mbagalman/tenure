# Installation

## Requirements

- Python **3.10 or newer**
- The core runtime dependencies are installed automatically: `pandas`, `numpy`,
  [`lifelines`](https://lifelines.readthedocs.io/), and `matplotlib`.

Tenure is **DataFrame-pure**: it takes a pandas DataFrame in and returns tidy, backend-neutral
result objects. You own reading and writing your own data.

## Install

!!! note "PyPI name not yet final"
    Tenure is pre-1.0 and not yet published to PyPI under a final distribution name. Until then,
    install from source.

From source (current recommended path):

```bash
git clone https://github.com/mbagalman/tenure.git
cd tenure
pip install -e .
```

Once published, installation will be the usual:

```bash
pip install tenure   # name TBD
```

## Optional dependency groups

```bash
pip install -e ".[dev]"    # pytest + ruff, for running the test suite and linter
pip install -e ".[docs]"   # mkdocs-material + mkdocstrings, to build this documentation
```

## Verify your install

```python
import tenure
print(tenure.__version__)            # 0.5.0

# Run the built-in demo end to end -- no data of your own needed.
result = tenure.naive_vs_corrected_demo()
print(result["ltv_dollar_diff"])     # 10.35
```

If that prints without error, you are ready for the [quickstart](tutorials/quickstart.md).

## Building these docs locally

```bash
pip install -e ".[docs]"
mkdocs serve        # live-reloading preview at http://127.0.0.1:8000
mkdocs build        # static site into ./site
```
