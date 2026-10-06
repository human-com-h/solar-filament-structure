# Output identities and missingness

`MASK_MANIFEST.json` binds run, method, seed, mode, threshold, split, full observation list, exported-file hash, tensor hash and each native mask/image hash. Native PNGs are single-channel binary 0/255. The scoring entry rejects a different operating point, model identity, missing/duplicate observations, changed mask bytes or nonbinary geometry. The current adapters define a portable manifest contract; archival manifests are retained separately and cannot be assumed to have the same fields.

Target NPZ files contain `mask`, `srl`, `labels`, `weights`, `offset` at 1024×1024. Their CSV manifest binds sample IDs to observations and frozen split rows. Label values preserve annotation-order ownership. Files exported with a sample filter are only that sample's target check; component scoring requires the entire internal-test set and verifies its 101/157/11 cohort.

Segmentation outputs use `physical_observation_id`, `radius`, run/mode/threshold and the frozen seven metric names. Per-copy and spine rows retain the original evaluator's field names. Cohort summaries average observations, not all copies as independent observations. Component outputs separately use `component_recall` and `pixel_recall` at training-pixel radii 0/1/2.

Axis outputs have reference, copy, observation and seed layers. Per-reference rows include match status, predicted/reference length and signed/absolute error fields when successful. Failed length fields are empty/None with `length_status=FAIL_NA`. Observation macro success is distinct from total successful-reference fractions and candidate yield. `successful_observations`, `successful_copies` and `successful_references` remain separate counts. Length averages exclude failures, and common-success statistics identify their reference intersection explicitly.

| Status | Meaning | Numerical handling |
|---|---|---|
| `OK` / `SUCCESS` | Measured operating point / qualified one-to-one axis | Retain actual values, including genuine zero scores |
| `FAIL` with `FAIL_NA` lengths | Evaluated axis recovery failed | Success contribution is zero; lengths remain missing |
| `INFEASIBLE_NA` | No nonempty validation threshold satisfies the locked floor | Planned rows are retained; no mask or substituted fixed-threshold score |
| Not assessed / not run | Endpoint or engineering execution was not performed | Describe explicitly; no fabricated zero or inferred performance |
| Technical error | Missing files, corrupt identities, dependency or runtime failure | Command fails; never convert it to scientific NA |

Original CSVs can spell missing numbers as `NA`; portable newly generated CSVs serialize None as empty fields. Explicit status fields disambiguate this representation. The replay routes use their unchanged original NA conventions and comparison checks.

Current manuscript tables label infeasible selected entries as `Infeasible`; figures use an em dash, with their meaning explained in the captions. The plotting adapter applies the original v5 display-only AST transform and removes the selected-soft-clDice footer. Neither change replaces scientific CSV states or measured fixed-mode zeros. A fixed-mode dash for undefined length remains distinct from selected threshold infeasibility.

Reference-mask calibration, copy-pair agreement, candidate yield and common-success lengths use their own denominators. They are not interchangeable endpoints. Bootstrap CI rows, three-seed sample SD and descriptive means remain in separate fields and tables. Negative paired effects, failed modes and year/group sensitivity rows are retained.
