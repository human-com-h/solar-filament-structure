# External inference exports

No weights or optimizer checkpoints are included in the source tree. The separate `solar-filament-weights.zip` contains 24 inference state-dicts in one `models/` directory. Its size is 606,442,575 bytes and its SHA-256 is `77cb57076377cbdd14c85e67eb0112e158a0f85f7bedfd2bf44a2af0bb4b5d86`. Each model preserves the original export bytes.

Extract the archive once to `weights-root`, yielding `weights-root/models/<run>.pt`. `configs/paper24/models.json` records each member/export/tensor identity. File and tensor hashes are checked before strict loading. A source checkpoint hash is distinct from the inference export hash. Weight distribution terms are separate from the MIT code license. See [INPUTS.md](../docs/INPUTS.md) for access and [TRAINING.md](../docs/TRAINING.md) for the separate full-state recovery requirement.
