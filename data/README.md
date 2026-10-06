# External observations and annotations

MAGFiLO source data are available from [Harvard Dataverse](https://doi.org/10.7910/DVN/J6JNVK), with direct downloads for the [v1.0 annotations](https://dataverse.harvard.edu/api/access/datafile/10407987) and [observation URL index](https://dataverse.harvard.edu/api/access/datafile/10409204). No actual data are included in this code directory.

Set `--data-root` to the study input tree containing `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` and `train/train_images/`. Preserve filenames and annotation/observation IDs. The fixed manifest, QC rows and all image hashes are obtained from the external accepted evidence.

Run `check-data` before data-dependent execution. Annotation SHA-256 is `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`. The public v1.0 JSON has a different identity from the study annotation version. Access and arrangement details are in [INPUTS.md](../docs/INPUTS.md); provider metadata is in `provenance/SOURCE_LICENSE_METADATA.json`.
