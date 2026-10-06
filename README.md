# Structural supervision and solar filament axis extraction

Code and reproduction workflows for **Evaluating structural supervision and main-axis extraction for solar filament segmentation**, by Zida Hao and Yubo Li. The repository includes the eight-method, three-seed implementations for training, inference, segmentation and main-axis evaluation, statistical analysis and figure generation. Configurations and source identities follow the scientific protocol documented in this repository.

The study compares segmentation coverage with recovery of annotation-defined main axes. Component balancing did not give consistent gains, and coverage and main-axis rankings differ. The region+clDice control changes both the regional objective and structural weight. Its results do not isolate a BCE effect. Temporal sensitivity and failed operating points remain part of the evidence. Residual branches are annotation-derived skeleton components; they do not establish recovery of physical solar branches.

## Methods and scope

| Internal ID | Paper name | Architecture / objective |
|---|---|---|
| `BCE_Dice` | BCE–Dice | U-Net; regional BCE–Dice |
| `SRL` | SRL | Same U-Net; regional loss plus skeleton recall |
| `SABR` | SABR | Same U-Net; skeleton recall and balanced residual-component recall |
| `ResidualPixel` | ResidualPixel | Same eligibility/fallback; residual pixel recall |
| `SRL_FG` | SRL–FG | Same eligibility/fallback; foreground recall |
| `FlatUNet_BCE` | Flat U-Net | Frozen Flat U-Net; BCE |
| `UNet_softDice_clDice` | soft-clDice | Original combined soft-Dice/soft-clDice objective |
| `UNet_region_clDice` | region+clDice | BCE–Dice plus structural soft-clDice; two-rank global reduction |

All eight methods have three indexed inference exports, with seeds `20260831`, `20260901`, `20260902`. The original soft-clDice selected operating points are `INFEASIBLE_NA` for all seeds. Fixed 0.5 scores remain measured, with zero successful axes and `FAIL_NA` lengths. MORDEN is outside this 24-model cohort: its resource failure is unmeasured performance. See [scientific protocol](docs/SCIENTIFIC_PROTOCOL.md) and [output definitions](docs/OUTPUTS.md).

## Start here

Python 3.12 is required. Keep the complete source tree. The entry script locates its own source directory and runs from any working directory; installing the package is optional. Frozen source files are archival implementations, and the public entry is `scripts/filament.py`.

```sh
python /path/to/repository/scripts/filament.py --help
python /path/to/repository/scripts/filament.py verify-release
python /path/to/repository/scripts/filament.py replay --models 21 --evidence-root /path/to/review_package --output-root /path/to/new/replay21
python /path/to/repository/scripts/filament.py replay --models 24 --evidence-root /path/to/review_package --output-root /path/to/new/replay24
```

Those two replay commands use only Python's standard library. They reconstruct accepted score summaries; they do not re-evaluate images or rerun bootstrap draws. Use new output directories. Every computation output must be outside this source tree.

For CPU target construction, evaluation and analysis:

```sh
python -m pip install -r /path/to/repository/requirements-cpu.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python /path/to/repository/scripts/filament.py validate --with-models --output-root /path/to/new/cpu_checks
```

Alternatively, from the repository use `python -m pip install -e ".[cpu,plots,tests]"` and install the CPU torch wheel separately. Editable installation keeps the frozen-source closure available. An isolated wheel containing only `src/` is not the distribution described here. GPU training uses a separate locked Linux runtime; do not substitute this CPU dependency set. See [installation](docs/INSTALLATION.md) and [training](docs/TRAINING.md).

Run `verify-release` on the pristine tree before an editable development installation, which can add packaging metadata to `src/`. Direct script execution and dependency-only installation avoid that added source-tree metadata.

## Inputs and full execution

The study uses MAGFiLO H-alpha images and filament annotations. The public source data are available from [MAGFiLO v1.0 on Harvard Dataverse](https://doi.org/10.7910/DVN/J6JNVK):

| Download | Content |
|---|---|
| [MAGFiLO annotation JSON](https://dataverse.harvard.edu/api/access/datafile/10407987) | `magfilo_2024_v1.0.json`, the public v1.0 annotations |
| [Observation URL index](https://dataverse.harvard.edu/api/access/datafile/10409204) | Source addresses for GONG H-alpha observations |
| [Dataset paper](https://doi.org/10.1038/s41597-024-03876-y) | Annotation and data preparation methods |

Reproducing this study requires the annotation version and 707 image files identified by the frozen manifest. The public v1.0 annotation file has a different file identity; use the recorded hashes to check the study inputs. Model weights, frozen result tables, masks and recovery state are separate from the source data. [Input identities and acquisition](docs/INPUTS.md) list the required files and explain how to arrange them.

Download the [24-model weight archive](https://github.com/human-com-h/solar-filament-structure/releases/download/V1.0.0/solar-filament-weights.zip) and [SHA-256 file](https://github.com/human-com-h/solar-filament-structure/releases/download/V1.0.0/solar-filament-weights.zip.sha256) from [Release V1.0.0](https://github.com/human-com-h/solar-filament-structure/releases/tag/V1.0.0). The separate `solar-filament-weights.zip` contains all 24 pretrained model exports in `models/<run>.pt`, together with `MODEL_INDEX.csv`, `README.md` and the MIT `LICENSE`. The model weights are licensed under MIT. [Weight package details](weights/README.md) record the archive size, SHA-256 and extraction instructions; each model retains its original exported bytes.

Four root options are available: `--data-root`, `--weights-root`, `--evidence-root`, `--output-root`. Copy `configs/paper24/paths.example.json` **outside** the repository, set absolute roots or paths relative to that configuration file, and pass `--config /path/to/paths.json`. Command-line roots take precedence. Annotation and image subpaths can be overridden with `--annotation` and `--images`; observation IDs and filenames remain bound to the frozen manifest.

```sh
python /path/to/repository/scripts/filament.py check-data --config /path/to/paths.json
python /path/to/repository/scripts/filament.py build-targets --config /path/to/paths.json --splits internal_test
python /path/to/repository/scripts/filament.py infer --config /path/to/paths.json --run SABR_seed20260831 --mode validation_selected --device cuda:0 --output-root /path/to/new/sabr_masks
python /path/to/repository/scripts/filament.py score-segmentation --config /path/to/paths.json --mask-manifest /path/to/new/sabr_masks/MASK_MANIFEST.json --output-root /path/to/new/sabr_segmentation
python /path/to/repository/scripts/filament.py score-axes --config /path/to/paths.json --mask-manifest /path/to/new/sabr_masks/MASK_MANIFEST.json --output-root /path/to/new/sabr_axes
```

Use separate output directories for each command. The complete [command reference](docs/COMMANDS.md) covers all entries, including independent validation-threshold reconstruction, component recall, five-method and 21/24-model statistics, S11, full diagnostics and all figure families. [Reproduction paths](docs/REPRODUCTION.md) distinguish score replay, mask rescoring, inference and training.

## Source organization and provenance

`src/filament_structure/` supplies root resolution, input checks and execution adapters. `frozen_sources/` contains the accepted scientific implementations and licensed vendor sources. Historical workspace and display names use canonical names; numerical algorithms, model identities and scientific settings are preserved. `configs/paper24/` supplies the exact model/threshold mapping and protocol. `analysis/README.md` maps analysis functions to their sources. `provenance/SOURCE_MAPPING.csv` records canonical source paths, original SHA-256 identities, release destinations and adaptation status. `FROZEN_SOURCE_IDENTITIES.json` distinguishes original hashes from hashes of adapted copies. `provenance/RELEASE_FILES_SHA256.csv` covers the complete file set, excluding its own hash.

No losses, labels, threshold rankings, axis matching criteria or statistical estimators were harmonized across sources. The staged baseline code removes eager imports for the already-excluded MORDEN method. Execution locks identify the staged layout separately from archival locks. All adaptations and their verification are described in [adaptations](docs/ADAPTATIONS.md).

## Verification and limitations

The [local verification record](docs/VALIDATION.md) reports executed CPU paths, observed tolerances, input identity checks and unexecuted GPU paths. Synthetic tests are engineering checks and never paper evidence. Full numerical GPU training and bitwise reproduction of newly trained weights are not claimed.

The repository does not contain raw data or the separate 24-export weight archive. Frozen evidence tables, mask exports, 170 original test labels, historical recovery state and external figure QA tools are also external inputs. The 24 model exports have author-approved public distribution under MIT. Data and other derived evidence retain their own access and distribution terms. The code is fully implemented; complete data-dependent reproduction still requires these inputs.

## Citation and licenses

If you use this code, please cite:

> Hao, Z., and Li, Y. *Evaluating structural supervision and main-axis extraction for solar filament segmentation*. Manuscript.

`CITATION.cff` provides the software record and preferred manuscript citation. A BibTeX entry and instructions for updating the published reference are in [citation guidance](docs/CITATION.md). When using MAGFiLO, also cite the [dataset](https://doi.org/10.7910/DVN/J6JNVK) and its [data descriptor](https://doi.org/10.1038/s41597-024-03876-y).

The authors' code, accompanying documentation and 24 model weight exports are licensed under the [MIT License](LICENSE). The separate weight archive includes its own copy of `LICENSE`. Flat U-Net and clDice retain their original MIT licenses in `frozen_sources/vendor/`. MAGFiLO v1.0 has its own [CC BY-NC 4.0 terms](https://creativecommons.org/licenses/by-nc/4.0/) in the Dataverse record; the MIT license does not cover datasets or other derived evidence. See [third-party notices](THIRD_PARTY_NOTICES.md).
