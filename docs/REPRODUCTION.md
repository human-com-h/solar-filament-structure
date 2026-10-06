# Four reproduction paths

| Path | What is recomputed | Inputs | Verification boundary |
|---|---|---|---|
| CPU score replay | Accepted per-reference/per-copy/per-observation summaries and NA rows | External accepted evidence | No new image evaluation or bootstrap draws |
| CPU frozen statistics | Original 100,000-draw intervals, paired effects, year/LOGO and common-success lengths | Accepted tables or original receipt sets | Source-specific RNG and denominator rules preserved |
| Mask rescoring | Native segmentation and exact mask-only axes from supplied binary masks | Exact masks, references, labels/annotations | Independent of weight inference; image-specific geometry is recomputed |
| Inference / retraining | New outputs from locked exports / new optimization trajectory | Exact permitted images, exports / training runtime and recovery state | Inference model identity is checked; retraining need not produce identical bytes |

`replay --models 21` runs the original standard-library replay and checks nine application tables against tolerance 5e-14. `replay --models 24` runs the original added-control replay and checks three combined tables against tolerance 1e-12. Both retain original NA checks. `statistics --arm original21` refers to application statistics for the original cohort; `segmentation21` separately redoes its accepted segmentation bootstrap. `region24` redoes the region-control statistics, including separate +7/+71 offsets. The original-five receipt routes preserve the segmentation and component bootstrap implementations and require the external receipt root.

Paper-cohort scoring uses only full frozen splits; there is no random pilot masquerading as full evaluation. The synthetic example and tests are explicitly engineering fixtures. A full new inference run would use all observations of the selected validation or internal-test split and write its input/mask hashes. Threshold reconstruction is validation-only and must match the locked operating point.

Figure families retain the original inputs and styles: segmentation produces Figures 1–3 and S1–S4; application produces Figures 4–5 and S5–S9; control produces Figure 7; cases produces Figure 6. They export PDF, SVG, 600-dpi PNG/TIFF, source records and QA reports. Figure 6 requires legal raw inputs and full diagnostics, applies the original retrospective median-stratum rule and extracts whole-image axes before display crops. Outputs are kept outside the source directory.

Supplement S11 uses the original target census and consolidation scripts. Its default census requires the 170 original NPZ test labels for exact historical-label parity. Consolidation retains all eight-method coverage rows and the separate original-five component endpoint, temporal composition and sensitivity denominators. Target arrays, score rows and observation-derived figures are external distribution items.

The [local validation report](VALIDATION.md) is the authority for what actually ran. A syntax check, missing-input rejection or CPU loss test cannot be described as a completed GPU or full-image reproduction.
