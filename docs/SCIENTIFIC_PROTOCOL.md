# Frozen scientific behavior

## Data, sampling and targets

Temporal groups are transitive consecutive-date components with gaps of at most 14 calendar days, combining stations and annotation copies. Use the accepted manifest; never regenerate a random split. The partitions contain 492/107/108 observations, 812/172/170 copies and 34/9/13 temporal groups for train/validation/internal test. Input IDs retain the original file mapping.

Training uses grayscale bilinear resize to 1024×1024. Each epoch samples one copy per physical observation with the original deterministic choice and batch ordering. Full target export can include all copies; the train/validation cache excludes test data. Native inference thresholds the 1024 probability grid first and restores the binary mask by nearest-neighbour resize.

The SRL target is the union-mask skeleton dilated by `diamond(2)` and intersected with foreground. Residual components use the original manual endpoint mapping, anchored shortest main path, path rejection, `diamond(1)` path neighbourhood removal and connectivity. The minimum five-pixel rule is applied before annotation-order overlap ownership. A component reduced below five pixels by ownership remains retained. No second size filter is added. Missing or rejected paths contribute no branch components; target validity and SRL fallback are unchanged. QC exclusions apply through the exact exclusion CSV. Branches are annotation-derived residual skeleton components, not physical feature labels.

## Objectives

Regional BCE–Dice is `0.5*mean BCE + 0.5*(1-mean per-sample soft Dice)`, with `eps=1e-6`. For the original five methods, structural weight is 0.05. SRL recall excludes empty structural targets from its batch mean. SABR uses `0.75*SRL + 0.25*equal-component recall` when residual and SRL targets are valid; otherwise it falls back to SRL. ResidualPixel and SRL–FG replace only the 0.25 special term and share SABR eligibility/fallback. The weight-map and offset implementation exactly represents the original component average. Empty SRL batches add differentiable zero structure loss.

Flat U-Net uses the frozen architecture and BCE. Original soft-clDice uses the vendor combined `soft_dice_cldice` with alpha 0.5, smooth 1, `exclude_background=False` and effective ten-iteration soft skeletons. Its global sufficient statistics are not the per-sample regional Dice used above. Region+clDice adds weight 0.05 times the structural-only `soft_cldice` term to regional BCE–Dice. It combines the regional objective and structural weighting; it does not isolate BCE.

Hard-mask clDice is an evaluation statistic based on binary-mask skeleton precision/recall. It is neither the combined differentiable training objective nor the structural-only differentiable term. The frozen code preserves each operation and its smoothing/empty convention.

## Training and thresholds

Three fixed seeds use float32, batch 2, AdamW at learning rate 1e-4 and weight decay 1e-5, no augmentation, AMP or scheduler, at most 150 epochs, patience 20 and min_delta 1e-6. Best checkpoints maximize observation-macro validation Dice at fixed 0.5. Flat seed 20260901 accepted continuation from epoch 92 to epoch 150, best epoch 137; this is provenance, not a request to manufacture an epoch-92 checkpoint.

Region+clDice uses two T4 GPU ranks, one image per rank, SyncBatchNorm and an autograd-aware SUM reduction of global sufficient statistics. Averaging two independently evaluated ratio losses is a different objective and is not an accepted replacement. CPU model checks use a single process only to test imports and algebra; genuine two-rank Gloo checks test the reduction and full-state recovery without claiming GPU verification.

The 39 thresholds are the exact unique rounded grid `np.r_[.01, np.arange(.05,.951,.025), .5, .99]`, cast to float32. Same-seed BCE–Dice precision floors are 0.6646950447657447, 0.6997871087918596 and 0.6910794546396283. A candidate must meet the floor and have nonempty foreground. Selection ranks lexicographically by MSC, negative MSGR, Dice, then threshold. Formal inference reads the locked selection. Independent reconstruction uses validation only and reports whether it equals the lock; test performance never selects thresholds.

## Evaluation and axis recovery

Dice, IoU, precision, recall, hard-mask clDice, MSC and MSGR use the unchanged evaluator. Segmentation spine samples use 0.5-pixel spacing and original rounding, with native-pixel radii 1/3/5. Copy-level results are aggregated within observations before cohort averages. Empty foreground and denominator conventions remain the original code's conventions, including its frozen EDT behavior.

Residual component recall uses the training grid and disk neighbourhood radii 0/1/2. Its cohort is independently fixed at 101 observations, 157 valid copies, 11 time groups, 1,934 components and 17,019 pixels. It was assessed for the original five methods only.

Axes are extracted from masks alone: skeleton components, native geometric edge weights, exact shortest-path diameter, original endpoint handling and 16-source Dijkstra chunks with deterministic ties. References do not assist extraction. No closing, bridge, pruning, approximate diameter or widened acceptance criterion is added. Axis matching uses continuous unrounded 0.5-pixel resampling, at least 90% coverage in both directions within three native pixels, maximum-cardinality one-to-one matching and the original quality priority.

All 1,368 references across 170 copies, 108 observations and 13 groups enter success-rate accounting. Macro success is computed reference→copy→observation. Candidate output, matched-reference counts, observation success and successful-length means have distinct denominators. Lengths are conditional on success; common-success comparisons use the intersection. Failed lengths are `FAIL_NA`, not zero errors.

## Statistics and interpretation

The original bootstrap uses 100,000 multinomial draws in 100 batches of 1,000, `numpy.default_rng(seed+71)`, lexical string group sorting, all observations carried with sampled groups, and observation-weighted estimates. The region analysis uses offset +7 for segmentation and +71 for application. These source-specific rules remain distinct. Directions are SABR-minus-comparator for the original analysis and region+clDice-minus-comparator for the added control. Native code retains year and leave-one-group-out analyses, negative effects and all planned NA rows.

Three-seed sample SD is descriptive and separate from within-seed bootstrap intervals. Intervals are not averaged. No length CI or new confirmatory test is introduced. Reference-mask and annotation-copy diagnostics are calibration diagnostics, not independent proof of scientific utility. Case selection is retrospective illustration with the original declared median-stratum rules.
