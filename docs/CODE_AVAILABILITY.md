# Code and external artifacts

This repository contains the implementation and reproduction interfaces for the eight-method, three-seed study described in *Evaluating structural supervision and main-axis extraction for solar filament segmentation*. The authors' code and accompanying documentation use the MIT License. Third-party code retains its original copyright and license notices.

The source includes model and threshold mappings, training and inference entries, segmentation and main-axis evaluation, frozen summary replay, statistical analysis, target-quality diagnostics, figure-generation code, provenance and verification records. Configurations and source identities follow the documented scientific protocol.

MAGFiLO source data are available through [Harvard Dataverse](https://doi.org/10.7910/DVN/J6JNVK), including [public v1.0 annotations](https://dataverse.harvard.edu/api/access/datafile/10407987) and the [observation URL index](https://dataverse.harvard.edu/api/access/datafile/10409204). Reproduction of the paper cohort requires the exact annotation version, observations and manifests documented in [INPUTS.md](INPUTS.md).

The 24 inference state-dicts, frozen evidence tables, original mask exports, original test labels and historical recovery state are separate artifacts. External figure quality-check tools are required by the original plotting workflow. Their input identities, expected layout and effects on execution are documented in the input reference. Executed paths and remaining runtime limits are reported in [VALIDATION.md](VALIDATION.md).
