# Analysis sources

The complete implementations are under `frozen_sources/analysis`, `frozen_sources/application`, `frozen_sources/s11`, `frozen_sources/diagnostics` and `frozen_sources/figures`, with portable invocation in `src/filament_structure/analysis.py` and `plotting.py`. This directory avoids duplicating those sources.

`statistics` exposes original-five segmentation/component receipts, original-21 application tables, original-21 segmentation bootstrap and region-control statistics as separate arms. `replay` exposes the two original standard-library replay scripts. `s11` binds the original census/consolidation paths. `diagnostics --action full` retains reference masks, copy pairs, candidate output, common-success length and geometric failure categories. `plot` binds the four accepted figure families, including declared retrospective case selection.

See `docs/COMMANDS.md` for executable commands and `docs/INPUTS.md` for the external dependency closure. Result CSVs and images are written outside this source directory.
