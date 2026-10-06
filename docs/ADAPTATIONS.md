# Execution adaptations

`FROZEN_SOURCE_IDENTITIES.json` records both original SHA-256 identities and the current hashes of public copies. Its `byte_identical_to_original` field distinguishes unchanged files from copies with canonical path or display names. Original source bytes and raw path bindings are retained outside the repository. New adapter code is identified separately in `SOURCE_MAPPING.csv`. Every released file, including documentation and configuration, is covered by the final checksum manifest.

| Adaptation | Reason | Scientific boundary / validation |
|---|---|---|
| Four roots and path traversal checks | Remove reliance on the original drive and working directory | Original IDs/filenames and input SHA checks retained; entries exercised from another directory |
| Canonical package, entry and workspace names | Provide stable names independent of manuscript administration | Package `filament_structure`, entry `filament.py`, workspace component `project`; an existing evidence workspace is discovered by its scientific directory layout, with ambiguity rejected and all input hashes retained |
| AST loading of selected original definitions | Avoid historical top-level path writes, torch imports in CPU scoring and manuscript side effects | Function/class bodies compiled unchanged; frozen file SHA identities checked; synthetic geometry/targets and frozen output parity checked |
| Model factory binding | Expose the eight accepted methods without obsolete imports | Exact U-Net/Flat/vendor implementations; parameter count, all losses and finite gradients checked |
| Staged baseline import removal | Original module eagerly imports an already-excluded MORDEN dependency | Remove only its sys.path entry and two imports; accepted CONFIG, losses and factory bodies unchanged; staged identity explicitly differs from archival identity |
| New execution code locks | Layout/vendor closure differs from archival full tree | Locks describe staged bytes; never overwrite archival locks or reuse incompatible archival recovery gates |
| Native inference and portable mask manifest | Bind new inference outputs to the model index and native masks | Exact grayscale/1024/threshold/nearest sequence retained; export/tensor hashes checked; no test selection |
| Component recall adapter | Expose the original disk-max neighbour/count computation | Radii and fixed cohort retained; independent original-function parity test |
| Statistics path binding | Historical outputs embed original absolute locations | Original source functions and RNG retained; only actual input closure hash checks resolve new paths; archived acceptance is input evidence, not a newly claimed acceptance audit |
| S11 compatibility layout | Original census expects test NPZ labels in the old tree | Copy external labels only to output; retain census and consolidation bodies; disable manuscript-backup operation only; all nine generated S11 CSVs matched original bytes |
| Full diagnostic compatibility layout | Frozen main expects a fixed workspace namespace | Verify exact frozen input hashes and original mask hashes; old geometry rows exactly replayed; no new bootstrap or thresholds |
| Figure root injection | Original plotting scripts write/copy historical manuscript files | Keep plot artists, values, size, fonts and exports; omit manuscript-copy setup; alignment/collision/text/source and rendered inspection recorded |
| External figure QA path | Replace a machine-specific tool directory with `external_figure_qa` in retained setup code | Public plots use the explicit `--qa-tools-root` argument and retain the pinned external tool hashes; plotting definitions and scientific data are unchanged |
| Current v5 display transform | Match current manuscript figure labels and captions | Original v5 DisplayOnly class changes text to an em dash and removes the selected-soft footer; scientific source rows and missingness states remain unchanged |
| Case preselection for dependency copying | Avoid copying all raw images into compatibility output | Same original eligibility/median ordering and stratum ties, then unchanged original main; only three selected legal images and masks copied outside repository |

Losses, labels, reductions, interpolation, thresholds, matching criteria and estimators retain their original behavior. Public copies use canonical workspace and display names. The two standalone replay entries also import the shared path resolver; their table reconstruction functions remain unchanged apart from canonical strings. Original archives retain their original bytes. Historical excluded experimental branches in retained sources do not expand the accepted public command set.

The compact reference-diagnostic command supplies the same exact axis geometry with a smaller output interface. Full original candidate/failure/common-success reporting is available separately with its input freeze. The original-five receipt statistics remain separate from the accepted-table 21/24 routes because their raw receipts are not in the review subset.
# Git clone integrity checks

Source inventories and `verify-release` exclude only the root `.git` directory or file. This allows checks after cloning or initializing a repository while retaining detection of nested repositories and unexpected source-tree files. No scientific source or experiment identity is changed by this adjustment.

