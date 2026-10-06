# External inference exports

The separate `solar-filament-weights.zip` contains 24 inference state-dicts in one `models/` directory. The model exports and accompanying documentation are licensed under MIT, copyright 2026 Zida Hao and Yubo Li. The archive includes `LICENSE`, `MODEL_INDEX.csv` and `README.md`. Each model preserves the original export bytes.

Download the [24-model weight archive](https://github.com/human-com-h/solar-filament-structure/releases/download/V1.0.0/solar-filament-weights.zip) and its [SHA-256 file](https://github.com/human-com-h/solar-filament-structure/releases/download/V1.0.0/solar-filament-weights.zip.sha256) from [Release V1.0.0](https://github.com/human-com-h/solar-filament-structure/releases/tag/V1.0.0). Verify the archive against the identity listed below before extraction.

| Archive identity | Value |
|---|---|
| File | `solar-filament-weights.zip` |
| Size | 606,443,387 bytes |
| SHA-256 | `ce70dd1d3743b05d9db03b92742b22ff461a831f3d2e57931d8a7e8d7a50407c` |
| Model exports | 24 |
| License | MIT; full text in the archive's `LICENSE` |

Extract the archive once to `weights-root`, yielding `weights-root/models/<run>.pt`. `configs/paper24/models.json` records each member/export/tensor identity. File and tensor hashes are checked before strict loading. A source checkpoint hash is distinct from the inference export hash.

Optimizer checkpoints are separate from these inference exports. Dataset and evidence materials retain their own terms. See [INPUTS.md](../docs/INPUTS.md) for access and [TRAINING.md](../docs/TRAINING.md) for the full-state recovery requirement.
