# Local verification

This record describes local execution on 5–6 October 2026 and is bound to the current `paper-v5` manuscript. No GPU training or new paper experiment was performed. Verification outputs, real data, labels, figures, masks and weight copies are outside the source tree. Machine-readable summaries are in `provenance/LOCAL_VALIDATION.json` and its supporting QA metadata.

The current copies with canonical names were independently checked again. `provenance/NORMALIZATION_VALIDATION.json` records 60 scientific Python files with identical algorithms and numeric constants after the documented string adaptations, 21/24-model replay, all eight CPU model/loss/gradient paths, strict loading of the 24 unchanged exports from the new archive, all 18 public command helps and 26 interface checks, the complete statistics/S11/diagnostic workflows, full native-mask rescoring, synthetic region recovery and the scientific-contract test. All 66 scientific CSV outputs are byte-identical to the previous verified outputs, and all 16 figure source files match the current manuscript. Figure alignment, collision and minimum text-size checks also passed. These repeated checks do not establish any of the unexecuted GPU paths listed below.

## Executed scientific and engineering checks

| Path | Observed result |
|---|---|
| Original21 standard-library replay | PASS, nine tables; 15,163 retained package members and 4,212 receipts verified; maximum absolute difference `3.552713678800501e-15`, within original `5e-14` tolerance; original NA checks retained |
| Full24 standard-library replay | PASS, three tables; maximum absolute difference `3.552713678800501e-15`, within original `1e-12` tolerance; selected infeasibility unchanged |
| Frozen statistics, S11 and complete calibration diagnostics | 66 CSVs compared in full, including 100,000-draw original-five/21/region statistics, sensitivities, common-success and failure rows; all byte-identical, maximum numeric difference 0 |
| Exact raw input check | 707 image SHA/byte identities, annotation hash, 1,154 manifest rows and QC identities checked |
| S11 target reconstruction | All 1,154 targets and 170 original test labels have zero label differences; original 101/157/11 cohort, 1,934 components and 17,019 pixels reconstructed |
| Full frozen mask diagnostics | 972 masks verified; 12,312 old model/reference rows exactly replayed; 170 reference-mask copies, 1,368 references, 84 copy pairs/40 observations, 42 mode rows including 3 infeasible rows, 36 paired-length comparisons |
| Public native-mask rescoring | BCE_Dice seed20260831 selected, full 108 observations: 324 segmentation radius rows, 1,368 references, 170 copies and 108 observation axis rows; all compared scientific values/statuses match exactly, maximum difference 0; this was mask rescoring, not new inference |
| Eight models/losses | Actual `[2,1,64,64]` CPU forward/loss/backward for all methods, exact parameter counts and finite gradients; empty/fallback and ten-iteration skeleton checks passed |
| All24 inference exports | Companion ZIP SHA matches; all 24 files loaded strictly and their export/tensor SHA identities matched; no full-split inference performed |
| Structural/geometry behavior | Exact axis self-tests; bidirectional overlong-candidate rejection; deterministic loop ties; macro .5 versus pooled .25 fixture; failed lengths remain missing; preownership 5→3 pixel component retention; endpoint/rejection/residual targets; component weighting difference `3.971816764369862e-9`; independent original radius-loop parity at 0/1/2 including boundary pixels |
| Public interfaces | All 18 command helps, 26 interface checks and 40 documented command occurrences checked from an outside working directory; relative config roots, missing-input messages, CPU training/preflight rejection and original target label parity passed; selected NA retained 1,368 reference and 324 segmentation rows |
| Meaningful pytest | One aggregate scientific-contract test passed; its assertions execute the behavior checks above rather than a stub or import-only test |
| Training source stage | Complete region/baseline dependency staging passed; repeated baseline stage passed without overwriting a different source |
| Original region CPU smoke/recovery | Loss/value/gradient equivalence, sampling parity, model/optimizer/RNG/loss recovery, transactional archive and gate rejection checks passed on synthetic data |
| Figure families | Figures 1–7 and S1–S9 generated from original sources and legal external inputs; source rows, alignment, collision and 5-point minimum text checks inspected; current v5 display transform retains scientific source bytes |

The CPU runtime used Python 3.12.9, NumPy 2.4.6, SciPy 1.15.2, scikit-image 0.25.0, OpenCV 4.11.0, Pillow 11.1.0 and torch 2.10.0+cpu. It is distinct from the archived GPU runtime. The local CPU torch installation remains outside the repository.

## Limits and recorded failures

The original two-process Gloo CPU smoke failed at process-group construction with `makeDeviceForHostname(): unsupported gloo device` in the current Windows torch build. Retrying the available UV transport setting yielded the same device error. The source and global-loss algorithm were not changed to obtain a passing result. Two-rank parity/recovery is therefore not verified in this local environment; the actual locked Linux/NCCL target preflight remains mandatory.

CPU initialization tensor hashes differed from archived GPU-environment initialization hashes for all three seeds. The original smoke records this difference; the required target initialization checks were retained. CPU loss/import checks and synthetic recovery do not establish exact target initialization, full 1024×1024 SyncBatchNorm behavior, target memory/speed, or two-T4 training. No target GPU gate was manufactured or marked passed.

Full-split inference, 39-point probability recomputation from images, GPU component inference and all formal retraining paths were not executed. Their implementations and exact configuration are present, and the CLI inputs and rejection paths were checked. Model loading and frozen-mask rescoring do not imply those unexecuted paths passed. Historical epoch-92 Flat recovery additionally requires the separate complete checkpoint.

The source-level plotting checker flags the unchanged control script's dynamic export loop and arithmetic figure width because it cannot infer them. Actual PDF/SVG/PNG/TIFF outputs exist at 174×66 mm and 600 raster dpi; rendered text/alignment/collision checks pass. The segmentation source's seed-aggregate warning is resolved by its displayed individual seeds/means and explicitly descriptive figures; no unimplemented interval is added. These static findings are recorded, not silently converted into source-check passes.

## Manuscript and immutable evidence

Current manuscript table numbers and displayed equations were compared with the saved v3 archive and are identical. The 24-row model index differs only in `submission_version`. The v5 display changes use the original `DisplayOnly` class: table infeasibility is explained as Infeasible and figure text uses an em dash; scientific CSV states remain unchanged. Current manuscript/PDF identities are in `MANUSCRIPT_BINDING.json`.

The original source archive, nested original21 ZIP and original weight archive match their frozen hashes. Public copies use canonical package, entry, workspace and external-tool names. Their catalog retains the original hashes and separately records current public-copy hashes; 50 of 76 indexed files remain byte-identical to the originals. The remaining copies have documented path, display, replay-import or execution-lock adaptations. Original source bytes and raw path bindings remain outside the repository. The mutable manuscript root was updated from v3 to v5 during earlier work; the current binding follows the author's instruction, and no manuscript or frozen experiment file was written by these adaptations.

The content audit checks actual files, types, sizes and uses; excluded material is absent from the source directory. A pristine-tree `verify-release` checks the complete file set and checksums. The authors selected MIT for their code and documentation on 6 October 2026. The public MAGFiLO file links were checked separately using Dataverse metadata and HTTP HEAD responses; no public data files were copied into this source directory. Access to study-specific external artifacts remains separate from code licensing and local verification.

The source file set excludes only `.git` at the repository root, so a Git clone can be verified without hashing local Git objects or its index. Nested Git directories and other extra files remain detectable. Original scientific identities are unaffected by this metadata exclusion.
