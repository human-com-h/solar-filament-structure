# Runtime selection

## Summary replay

Use Python 3.12 and the complete tree. No pip packages are required for `replay` or `verify-release`. The script adds its own `src/` path and suppresses bytecode writes. Do not copy just an entry script or run archival sources directly: their historical top-level paths are preserved as provenance.

## CPU analysis and model checks

`requirements-cpu.txt` records the tested numerical stack; model checks additionally require torch 2.10.0 from the CPU wheel index. The local validation used Python 3.12.9, NumPy 2.4.6, SciPy 1.15.2, scikit-image 0.25.0, OpenCV 4.11.0 and torch 2.10.0+cpu. These are engineering and analysis checks, not the archived GPU training environment.

Install into an isolated environment outside the source tree. Optional editable installation is supported by `pyproject.toml`. Its package version 0.1.0 identifies the Python adapter distribution. The source directory and `frozen_sources/` remain required together; environments and dependency binaries are installed separately.

For plot commands, separately obtain the original QA script closure and install its dependencies (including PyMuPDF). The adapter reports missing tools instead of silently skipping them. Tool identities and distribution boundary are recorded in the third-party notices.

## Frozen GPU runtime

The accepted region runtime spec requires Linux x86_64, Python 3.12.13, torch 2.10.0+cu128, CUDA 12.8, cuDNN 91002, NumPy 2.0.2, Pillow 11.3.0, SciPy 1.16.3 and scikit-image 0.25.2. Install `requirements-gpu.txt` and the exact CUDA torch wheel in that environment. The unchanged `RUNTIME_SPEC.json` records installer URLs and SHA-256; their future network availability is not guaranteed by this local delivery.

The archived `bootstrap_runtime.py` supports the original Kaggle Linux layout. It installs into a dedicated temporary runtime and never resets the session clock. A generic non-Kaggle machine should provision the same recorded runtime, then run the public preflight, which checks actual numerical versions and hardware. Region formal execution requires two T4s, SyncBatchNorm and NCCL; CPU/Gloo algebra tests cannot satisfy that GPU gate.
