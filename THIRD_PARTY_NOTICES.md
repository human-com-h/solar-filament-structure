# Third-party sources and distribution scope

## Flat U-Net

The frozen `fs_model.py` comes from the Flat U-Net source recorded in the accepted provenance (Zenodo record `14610155`, source archive MD5 `935f09ae6dfa5eefeef50ff623fec541`). The exact included file SHA-256 is `5ff2983e140b4286239e3142f73e852ecb2567d18b4b3c5738e0baaf92d0a27e`. Its original MIT license, copyright 2024 gaofei zhu, is retained at `frozen_sources/vendor/flat/LICENSE`. No model weights are included.

## clDice

The included combined objective and soft skeleton implementation are pinned by the accepted source records to `jocpae/clDice` commit `47d31a6cc4a8101b1ffe8052994821961e57af9f`. `frozen_sources/vendor/cldice/LICENSE` retains the original MIT terms and copyright. The exact file hashes are in `SOURCE_MAPPING.csv` and the unchanged baseline `SOURCE_PROVENANCE.json`. The effective soft-skeleton depth used here is ten iterations. The combined training objective, structural-only objective and hard-mask evaluation metric are distinct operations.

## Installed numerical dependencies

Python, PyTorch, NumPy, SciPy, scikit-image, Pillow, OpenCV, psutil, pandas, Matplotlib, PyMuPDF, pytest and packaging tools are installed separately from their respective distributions. Their licenses are governed by those distributions; none of their environments or binaries are copied here. CPU and GPU dependency sets are separated because the archived training environment is numerically locked.

## External figure QA

The original figure code imports `audit_panel_alignment.py` and `audit_figure_collisions.py`. Their source bundle has no verified redistribution grant in the material checked for this delivery. They are intentionally external and supplied via `--qa-tools-root`. Obtain the matching tools from the original author environment after clarifying their distribution terms. Their expected identities are recorded in `provenance/EXTERNAL_TOOL_IDENTITIES.json`. Do not replace these imports with unconditional passing checks.

## Data and authors' code

MAGFiLO v1.0 is distributed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) according to its [Harvard Dataverse record](https://doi.org/10.7910/DVN/J6JNVK). The verified public annotation and observation-index download links are listed in `docs/INPUTS.md`; their metadata are in `provenance/PUBLIC_DOWNLOAD_LINKS.json`. `SOURCE_LICENSE_METADATA.json` retains the provider metadata from the study's annotation version without changing its original bytes. Separately supplied study data, masks, tables and model weights retain their own distribution terms.

The authors' code and accompanying documentation are distributed under the root `LICENSE` (MIT), copyright 2026 Zida Hao and Yubo Li. The original Flat U-Net and clDice copyright and MIT license files remain intact. Other dependencies and separately obtained artifacts are governed by their respective licenses.
