---

## 2. `data/README.md`

```markdown
# data

This directory contains the **input data files** required for the project.

## Expected contents

This folder is intended to store the observational and auxiliary files needed to run the scripts, for example:

- data tables in `.txt`, `.dat`, or `.csv` format
- covariance matrices
- catalogues
- supporting files for the analysis

## Usage guideline

All scripts should read input data from this directory using paths defined relative to the project root. For example:

```python
DATA = ROOT / "data"