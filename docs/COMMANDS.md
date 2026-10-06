# Command reference

Invoke `python /absolute/path/to/repository/scripts/filament.py <command>`. Every command supports `--config` and the four root flags. `output-root` is required except for `verify-release`; it must be outside the repository. Roots listed below are required even when an operation uses only a subset of their files. A missing required root is an argument error; a missing file, identity mismatch, unsupported scientific mode or dependency failure returns a nonzero status with context. Original scientific assertions remain active. Use `--help` for exact choices.

| Command | Additional flags and inputs | Outputs / environment |
|---|---|---|
| `verify-release` | None | Full source file-set/hash and frozen-source checks; root `.git` metadata excluded, nested repositories rejected; standard library |
| `validate` | `--with-models` optional | `CPU_VALIDATION.json`; CPU scientific stack, torch only with models |
| `check-data` | data/evidence; `--annotation-only` optional | `DATA_CHECK.json`; exact mapping and identity checks |
| `build-targets` | data/evidence; `--splits train validation internal_test`, `--sample-id` optional | `targets/<sample>.npz`, `TARGET_MANIFEST.csv`; CPU |
| `stage-training` | evidence; `--method`, `--seed` | `execution_sources/` dependency closure and separate staged code lock; no training |
| `prepare-cache` | data/evidence; `--method`, `--cache /path/cache` | Original train/validation cache; test excluded; CPU stack plus torch |
| `train-preflight` | data/evidence/cache; Flat/soft or region method; baseline `--resource-record` | Actual GPU smoke reports and `preflight/RELEASE_GATE.json`; locked GPU runtime; no formal run |
| `train` | data/evidence/cache; any of eight `--method`, one fixed `--seed`; Flat/soft/region `--gate` | `training/<method>_seed<seed>/` checkpoints, status and histories; locked GPU runtime |
| `infer` | data/evidence/weights; `--run`, `--mode fixed_0.5\|validation_selected`, `--split validation\|internal_test`, `--device` | Binary native masks and `MASK_MANIFEST.json`; torch CPU or CUDA |
| `select-threshold` | data/evidence/weights; `--run`, `--device` | 39-point `validation_curves.csv`, `THRESHOLD_CHECK.json`; fails if selection differs from lock |
| `score-segmentation` | data/evidence; `--mask-manifest` | Per-copy, per-observation, spine and summary CSVs at native radii 1/3/5 |
| `score-axes` | evidence; `--mask-manifest` from internal_test | `per_reference.csv`, `per_copy.csv`, `per_observation.csv`, `per_seed.csv`, scope identity |
| `score-components` | data/evidence/weights; `--run`, `--targets-root`, `--device` | `branch_per_observation.csv`, radii 0/1/2; original five methods only |
| `replay` | evidence; `--models 21\|24` | Original `REPLAY_QA.json` and 9/3 reconstructed tables; standard library |
| `statistics` | evidence; `--arm original21\|segmentation21\|region24\|mechanism-region\|mechanism-components`; last two need `--receipts-root` | Exact original bootstrap, effects, year/LOGO and common-success tables; NumPy CPU |
| `s11` | evidence; `--action census` additionally data and `--labels-root`; or `--action consolidate` | Target-quality tables, denominator/QC/group indexes and sensitivity integration |
| `diagnostics` | data/evidence; `--action reference` or `full`; full adds masks-root and diagnostic-protocol | Reference masks/copy-pair summary, or full original candidate yield, common lengths, failure categories and receipts |
| `plot` | evidence; `--family segmentation\|application\|control\|cases`, `--qa-tools-root`; cases adds data/masks/diagnostics roots and selection-protocol | Original PDF/SVG/600-dpi PNG/TIFF plus source, alignment/collision reports |

`diagnostics --action reference` is a compact target-only route. `--action full` runs the full original diagnostic implementation and verifies the exact analysis input freeze. It also writes the reference-mask receipts needed by the case figure. These routes have different output filenames; use the full route for `plot --family cases`.

## Complete command examples

Replace `/repo`, `/inputs` and `/out` with real absolute paths. `/inputs/paths.json` contains the four roots. Each output directory below is distinct and outside `/repo`.

```sh
python /repo/scripts/filament.py build-targets --config /inputs/paths.json --splits internal_test --output-root /out/test_targets
python /repo/scripts/filament.py score-components --config /inputs/paths.json --run SABR_seed20260831 --targets-root /out/test_targets --device cuda:0 --output-root /out/component_scores
python /repo/scripts/filament.py select-threshold --config /inputs/paths.json --run SABR_seed20260831 --device cuda:0 --output-root /out/validation_threshold
python /repo/scripts/filament.py statistics --config /inputs/paths.json --arm original21 --output-root /out/application_bootstrap21
python /repo/scripts/filament.py statistics --config /inputs/paths.json --arm segmentation21 --output-root /out/segmentation_bootstrap21
python /repo/scripts/filament.py statistics --config /inputs/paths.json --arm region24 --output-root /out/region_bootstrap
python /repo/scripts/filament.py statistics --config /inputs/paths.json --arm mechanism-region --receipts-root /inputs/original_revision --output-root /out/mechanism_region
python /repo/scripts/filament.py statistics --config /inputs/paths.json --arm mechanism-components --receipts-root /inputs/original_revision --output-root /out/mechanism_components
python /repo/scripts/filament.py s11 --config /inputs/paths.json --action census --labels-root /inputs/test_labels --output-root /out/s11
python /repo/scripts/filament.py s11 --config /inputs/paths.json --action consolidate --output-root /out/s11
python /repo/scripts/filament.py diagnostics --config /inputs/paths.json --action reference --output-root /out/reference_diagnostic
python /repo/scripts/filament.py diagnostics --config /inputs/paths.json --action full --masks-root /inputs/mask_export --diagnostic-protocol /inputs/calibration/PROTOCOL_CN.md --output-root /out/full_diagnostic
python /repo/scripts/filament.py plot --config /inputs/paths.json --family segmentation --qa-tools-root /inputs/figure_qa --output-root /out/segmentation_figures
python /repo/scripts/filament.py plot --config /inputs/paths.json --family application --qa-tools-root /inputs/figure_qa --output-root /out/application_figures
python /repo/scripts/filament.py plot --config /inputs/paths.json --family control --qa-tools-root /inputs/figure_qa --output-root /out/control_figure
python /repo/scripts/filament.py plot --config /inputs/paths.json --family cases --qa-tools-root /inputs/figure_qa --masks-root /inputs/mask_export --diagnostics-root /out/full_diagnostic --selection-protocol /inputs/resolution/INTEGRATION_PROTOCOL_CN.md --output-root /out/case_figure
python /repo/examples/synthetic_axis.py --output-root /out/synthetic_example
```

The `s11` consolidation intentionally reuses its census output directory. Its frozen functions add their own tables and verify upstream frozen score parity. Replay and statistics otherwise require new/empty output directories. Repeated target/inference file writes are rejected. Do not use a result folder from a different model or source identity.

Training examples are in [TRAINING.md](TRAINING.md); all eight IDs are accepted by `train`. GPU preflight is an explicit engineering execution, not a shortcut around environment or recovery checks.
