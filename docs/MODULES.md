# Implementation coverage

| Scientific module | Public implementation | Frozen implementation |
|---|---|---|
| MAGFiLO loading, copy mapping, manifest/QC, identity | `data.inputs`, `check` | `mechanism/annotation_helpers.py`, `data_pipeline.py` |
| Union rasterization, grayscale/resizing, epoch copy sampling/cache | `data.targets`, `training.prepare_cache` | Original data pipeline and baseline/region data modules |
| Endpoint path, rejection, residual labels, ownership and balancing | `frozen.target_functions`, `data.targets` | `branch_reference.py`, `label_copy`, `balanced_weights` |
| Eight complete networks and objectives | `models.factory`, `loss`, `training.train` | U-Net, Flat, objectives/historical losses, vendor clDice and region global distributed math |
| Training, optimizer, early stopping, initialization, recovery | `training.stage`, `preflight`, `train` | Original trainer; baseline engine/runtime; region ddp_engine/runtime |
| Locked inference and validation grid reconstruction | `inference.infer`, `thresholds` | Frozen evaluator/threshold curve and indexed export identities |
| Segmentation/radius sensitivity and component recall | `evaluation.segmentation`, `components` | Original native evaluator and mechanism component evaluation |
| Exact mask-only axes and matching | `evaluation.axes`, `frozen.score_one` | `axis/axis_evaluator.py`, application `score_one` |
| 21/24 standard-library replay | `analysis.replay` | Unchanged replay21/replay24 sources |
| Original-five, 21-model and added-control statistics | `analysis.statistics` | Original bootstrap, summarize_mechanism_test, audit_and_analyze, application_statistics, region_postprocess |
| S11 census/consolidation | `analysis.s11` | `s11/compute_target_quality.py`, `consolidate_evidence.py` |
| Reference masks, copy pairs, yield, common lengths/failures | `analysis.diagnostics`, `full_diagnostics` | Original calibration functions/main and portable diagnostic adapter |
| Figures 1–7, S1–S9 and source exports | `plotting.figures`, `case_figure` | Four accepted figure-source scripts |

Public function definitions live in `src/filament_structure/`; `scripts/filament.py` dispatches all commands. `analysis/README.md` gives the analysis grouping. `examples/synthetic_axis.py` is an authored synthetic interface example, not a paper observation. `tests/test_scientific_contract.py` executes key scientific behavior checks and uses no private input.
