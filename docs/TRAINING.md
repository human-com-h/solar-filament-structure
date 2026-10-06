# Training and recovery

This preparation did not run GPU training. The eight training entries invoke the accepted engines with their real data, cache, model, optimizer, early stopping and recovery implementations. A recipient must provide the exact permitted inputs and actual target environment. Inference exports are not recovery checkpoints.

## Staging and cache

Use a separate output root per method family. `stage-training` copies the source closure to `output-root/execution_sources/`. It verifies frozen source identities, adds the exact external manifest/QC files and protocol, preserves scientific code and creates a code lock for this staged layout. The original archival locks remain in `frozen_sources/` and are not relabeled.

```sh
python /repo/scripts/filament.py stage-training --config /inputs/paths.json --method SABR --output-root /out/mechanism
python /repo/scripts/filament.py prepare-cache --config /inputs/paths.json --method SABR --cache /out/cache_mechanism --output-root /out/mechanism
python /repo/scripts/filament.py train --config /inputs/paths.json --method SABR --seed 20260831 --cache /out/cache_mechanism --output-root /out/mechanism
```

For `BCE_Dice`, `SRL`, `SABR`, `ResidualPixel`, `SRL_FG`, substitute that exact ID; each seed is explicitly selected. The original five-method trainer has no strong-baseline release-gate file. It preserves its own deterministic initialization, cache, data and resume assertions. The CLI rejects unavailable CUDA before training starts. `prepare-cache` uses only train and validation, through the unchanged original pipeline.

## Flat U-Net and original soft-clDice

Their accepted `FROZEN_DESIGN.json` contains these two measured methods. Archival `methods.py` still imports the earlier excluded MORDEN stack eagerly. The staging adapter removes only its path entry and two eager imports, and records the modified execution-copy identity. The factory branches and accepted method configuration remain unchanged. The discarded stack is not a dependency of either accepted method.

```sh
python /repo/scripts/filament.py prepare-cache --config /inputs/paths.json --method FlatUNet_BCE --cache /out/cache_baselines --output-root /out/baselines
python /repo/scripts/filament.py train-preflight --config /inputs/paths.json --method FlatUNet_BCE --seed 20260831 --cache /out/cache_baselines --resource-record /inputs/MORDEN_CHECKPOINT_EXCLUSION.json --output-root /out/baselines
python /repo/scripts/filament.py train --config /inputs/paths.json --method FlatUNet_BCE --seed 20260831 --cache /out/cache_baselines --gate /out/baselines/preflight/RELEASE_GATE.json --output-root /out/baselines
```

Substitute `UNet_softDice_clDice` for its training. The preflight invokes actual local and target engineering smoke runs, initialization checks, environment checks and original `release.py`. It generates a gate from evidence, never from invented passing flags. The resource failure JSON must match the checksum in the accepted design. It establishes MORDEN exclusion, not MORDEN performance. These GPU engineering steps are separate from a formal run and were not executed during this local delivery.

## Region+clDice

Provision the locked Linux runtime in [INSTALLATION.md](INSTALLATION.md), expose exactly two T4 GPUs, and use the same output root for staging, preflight and training. The original CPU and two-rank parity checks are run by the target preflight, followed by the actual target environment and DDP smoke checks.

```sh
python /repo/scripts/filament.py prepare-cache --config /inputs/paths.json --method UNet_region_clDice --cache /out/cache_region --output-root /out/region
python /repo/scripts/filament.py train-preflight --config /inputs/paths.json --method UNet_region_clDice --seed 20260831 --cache /out/cache_region --output-root /out/region
python /repo/scripts/filament.py train --config /inputs/paths.json --method UNet_region_clDice --seed 20260831 --cache /out/cache_region --gate /out/region/preflight/RELEASE_GATE.json --output-root /out/region
```

The public training entry launches `torch.distributed.run` with two ranks. The original `ddp_engine.py` preserves one sample per rank, global batch 2, synchronized BatchNorm, autograd-aware global sufficient-statistic reductions, exact historical sampling order, two-rank RNG state and recovery history. CPU, single-GPU or per-rank ratio averaging are not supported substitutes. The target gate checks actual code and numerical environment, so an archival gate cannot be reused with the staged code identity.

For CPU-only checks of the staged region source, run `local_smoke.py --output <stage>/evidence` and `cpu_ddp_smoke.py --output <stage>/evidence` from that staged directory. They exercise synthetic 64×64 data or toy distributed models and state recovery. They are not GPU/full-resolution validation. Windows Gloo device support can differ from Linux; record any transport failure without changing the reduction algorithm.

## Recovery and continuation

The original engines keep their transactional saves and automatic same-run recovery. Reuse the same method, seed, training directory, cache, staged code and exact environment to resume a paused run created by this layout. Their signatures reject code/environment drift. The strong-baseline/region recovery state includes optimizer, RNG, best model, epoch, stale counter and history; region state also includes RNG for each rank.

The accepted Flat seed 20260901 continued from epoch 92 and finished at epoch 150, best epoch 137. That historical record is retained in `configs/paper24/reproduction.json`. Reproducing that continuation requires the historical complete recovery state and exact original code/environment. Do not attempt to resume it using an inference state-dict or bypass an identity mismatch. No optimizer archive is promised by the 24-export companion.

All formal configurations retain batch 2, float32/no AMP, AdamW lr 1e-4 and weight decay 1e-5, maximum 150 epochs, patience 20, min_delta 1e-6 and fixed-0.5 validation macro-Dice selection. No scheduler or augmentation is added. Newly trained weights are not guaranteed byte-identical across environments. Validation-threshold selection remains separate from best-checkpoint selection and never uses test data.
