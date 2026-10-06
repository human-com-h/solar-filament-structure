# External inputs

MAGFiLO source data have public download links below. This code directory contains no actual observations, full annotation JSON, state-dicts, checkpoints, mask collections or frozen score tables. The study-specific annotation version, weight archive and evidence files are separate inputs; obtain the files matching the recorded identities before data-dependent execution.

## Four roots

| Root | Required content | Used by |
|---|---|---|
| `data-root` | MAGFiLO annotation version and JPEGs used in the study | Data checks, targets, cache, training, inference, segmentation/components, reference diagnostics and cases |
| `weights-root` | Verified exports under `models/<run>.pt` | Inference, independent threshold reconstruction, component scoring |
| `evidence-root` | Extracted accepted `review_package` with its transport and member indexes | Split/QC checks, references, replay, statistics, S11 and plots |
| `output-root` | New writable directory outside the repository | Every computing command |

The example JSON has null roots to avoid guessing machine locations. Relative roots resolve against the configuration file, not the process working directory. `--annotation` and `--images` override the two relative data paths.

## Data downloads and study cohort

| Resource | Link and content |
|---|---|
| MAGFiLO v1.0 record | [Harvard Dataverse](https://doi.org/10.7910/DVN/J6JNVK), version 1.0 |
| Public annotations | [Download JSON](https://dataverse.harvard.edu/api/access/datafile/10407987): `magfilo_2024_v1.0.json`, 60,602,626 bytes, recorded MD5 `bfa414567b4dfa2a6b4a9eccedb6aa20` |
| Observation addresses | [Download URL index](https://dataverse.harvard.edu/api/access/datafile/10409204): `[metadata]_active_observations_url.json`, 4,578,036 bytes, recorded MD5 `67c228c0409490f7acedc90740331f82` |
| Data descriptor | [Scientific Data article](https://doi.org/10.1038/s41597-024-03876-y) describing MAGFiLO |

The two direct download endpoints returned HTTP 200 with the expected filename and content length on 6 October 2026. File metadata and link checks are recorded in `provenance/PUBLIC_DOWNLOAD_LINKS.json`. The observation index supplies source URLs; it is not an image archive. The Dataverse v1.0 record assigns [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) to that public dataset.

The default annotation path is `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`, SHA-256 `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`. Images are under `train/train_images/`, named exactly as the frozen manifest. There are 707 observations and 1,154 annotation copies. `04_access/DATA_INPUT_INDEX.csv` in the external review package contains all 707 image identities. `check-data` checks every hash and byte size; `--annotation-only` explicitly omits the image check.

The study annotation file has a different identity from the public v1.0 JSON. The public download links identify the source release; paper reproduction still requires the exact annotation version and image files listed above. Keep the recorded filenames and observation IDs and run `check-data` to verify them. The provider metadata in `provenance/SOURCE_LICENSE_METADATA.json` is retained unchanged from the study annotation file. The authors' MIT code license does not assign terms to separate datasets, weights or derived evidence.

## Separate artifacts

Data and large artifacts are kept outside the code tree. The existing `data/` and `weights/` directories contain access instructions. A weight or evidence download must identify the corresponding archive size and SHA-256. Scientific evidence inputs have the extracted layout below; correspondence and other submission materials are not computation inputs.

## Frozen evidence layout

Preserve the complete extracted review-package structure and its transport indexes together. `project` below is the canonical workspace name used in this source tree. The resolver also accepts the original workspace folder by its scientific directory layout, without renaming or rewriting the supplied evidence:

```text
review_package/
  02_evidence/
    FROZEN_MEMBER_INDEX.json
    original21/
      PACKAGE_FILES_SHA256.csv
      payload/project/...
    region/workspace/project/...
  03_source_data/supplement_tables/...
  04_access/DATA_INPUT_INDEX.csv
```

The original21 replay verifies all 15,163 retained scientific members and 4,212 receipts. Partial score-only copies do not satisfy that command. Additional figure and S11 inputs are indexed in `provenance/EXTERNAL_INPUTS.csv`. The accepted manifests, locks, QC rows and reference axes remain external because their derivative distribution scope has not been established. Their absence blocks paper-cohort computation, but synthetic checks remain usable.

## Weights

The companion `solar-filament-weights.zip` is 606,442,575 bytes, SHA-256 `77cb57076377cbdd14c85e67eb0112e158a0f85f7bedfd2bf44a2af0bb4b5d86`. It contains all 24 exports directly in `models/`, plus a model index and an English README. Extract it once to `weights-root`, yielding `weights-root/models/<run>.pt`. Every model retains its original exported bytes. The CLI checks both file SHA-256 and sorted state-tensor identity and requires a strict model load. The MIT code license does not grant a license for these separate weights.

`configs/paper24/models.json` and `provenance/MODEL_EVIDENCE_INDEX_24.csv` distinguish source training-checkpoint, inference-export and tensor hashes. These 24 exports are inference state-dicts. They do not promise optimizer state, historical RNG state or a complete recovery archive. This repository currently supplies no weight download URL.

## Inputs omitted from the review subset

| Input / flag | Exact purpose and expected content | Consequence if absent |
|---|---|---|
| `--labels-root` | 170 original `<annotation_sample_id>.npz` test labels from `mechanism_branch_test/labels` | S11 cannot complete original-label parity; no passing label check is fabricated |
| `--receipts-root` | Original revision root containing `mechanism_locked_test/results` and `mechanism_branch_test`, including COMPLETE/signature files and per-run observation JSONs | Original-five segmentation/component bootstrap paths cannot be executed; 21/24 summary replay and accepted-table statistics still work |
| `--masks-root` | Original `mask_export` tree, its `MASK_MANIFESTS.json`, per-run manifests and binary PNGs | Full calibration diagnostics and Figure 6 cannot reconstruct mask-based results |
| `--diagnostic-protocol` | Exact `resolution_20261002/application_calibration/PROTOCOL_CN.md`, SHA-256 `7bb086b4c54804e3595afbfede45ffd231462e0ae05c5f870679906c12ecc44f` | Full diagnostic input freeze cannot be verified |
| `--diagnostics-root` | Completed full diagnostic output including reference-mask CSVs and receipts | Cases cannot verify reference-mask geometry |
| `--selection-protocol` | Original `resolution_20261002/INTEGRATION_PROTOCOL_CN.md` | Case-selection source identity cannot be recorded |
| `--resource-record` | Checksum-defined `MORDEN_CHECKPOINT_EXCLUSION.json` referenced by baseline FROZEN_DESIGN | Flat/soft target release gate cannot pass; MORDEN remains unmeasured |
| `--qa-tools-root` | Original figure QA scripts and their import closure; hashes in `EXTERNAL_TOOL_IDENTITIES.json` | Figure generation deliberately fails before exporting an unchecked result |
| Historical recovery archive | Exact `recovery.zip` or expanded state, identity and epoch files; complete optimizer, RNG, best model and history | Historical Flat continuation cannot be resumed from inference exports |

These are external dependencies, with recorded identities and verification results where applicable. They are not bundled in the source tree; obtain the matching files before running the relevant command.

Training source stages and compatibility inputs are created only under the selected external output root. Figures derived from real observations must also remain external until their distribution scope is settled.
