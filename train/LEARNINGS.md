# Local smoke status

The sandbox cannot communicate with the NVIDIA driver. No GRPO or SFT optimizer
step was run, and no trained adapter or GPU performance result is claimed.
Peak VRAM, seconds/step, episodes/hour, reward spread and actual LoRA change/reload
remain unmeasured. Reproduce the status with:

```bash
train/.venv/bin/python -m train.stack --check-tokenizer --output train/evidence/stack.json
```

The installed CUDA wheel has Blackwell build support, and the pinned TRL exposes
`environment_factory`. The probe exercises the pinned Qwen tokenizer's real tool
parser and assistant masks without downloading model weights. The reset text
needed compact public tables to meet the prompt budget on the larger smoke task;
the resulting token counts are in the same probe output.

SFT must tokenize before constructing the Arrow dataset: otherwise differing
source maps can acquire null entries. Explicit assistant masks survive the
pretokenized TRL path and exclude tool feedback. Tests also distinguish a failed
grader from a policy that earns zero, and keep diagnostic anchors smoke-only.
Reproduce the complete CPU suite with:

```bash
uv run --project train --locked pytest -c train/pyproject.toml train/tests -q
```

The smoke diagnostic selection is available, but the field directory for
`01addc2a86f1-s47` was missing at handoff. The GPU commands and prerequisites are
in [README.md](README.md). No field precompute, full training, native evaluation,
spend, publishing, branch creation or commit was performed by this WP.
