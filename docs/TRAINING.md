# Training

Use `--stage pretrain` for P0 → P1, then `--stage posttrain` for action alignment
→ joint fine-tuning. Each command launches its GPU workers and handles checkpoint
transitions automatically. P0 and P1 together form Stage 1. The recipe is installed with the package in
[`awm/configs/training.yaml`](../src/awm/configs/training.yaml).

| Phase | Optimizer updates | Global batch | Peak LR | Video sampling probability | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | --- |
| P0: bidirectional video adaptation | 2,000 | 64 | 1e-5 | 100% | DiT and video head |
| P1: causal video pretraining | 8,000 more | 64 | 1e-5 | 100% | DiT and video head |
| Action alignment | 2,500 | 64 | 1e-5 | 10% | Self-attention, action embedding and action head |
| Joint fine-tuning | 6,500 | 32 | 5e-6 | 20% | Entire DiT and both heads |

The VAE and text encoder are frozen. Training uses AdamW and BF16 autocast.
Global batch = processes × micro-batch × gradient accumulation. The launcher
calculates accumulation and rejects combinations that change the global batch.
Changing the number of GPUs therefore does not silently change this parameter.

Joint LR warms up for 500 updates, stays at 5e-6 through update 4,000,
decays with a cosine to 5e-7 at update 5,500, and stays there through 6,500.
P0 uses 200 warmup updates; P1 continues P0's optimizer state at peak LR
and continues sampling after the first 2,000 updates.
Alignment has constant LR. These last two scheduler details, AdamW betas
(0.9, 0.999), weight decay 0.01, epsilon 1e-8, gradient clipping at 1.0,
and text dropout 0.1 are implementation settings retained from the source
training recipe; the paper does not specify all of them.

## Installation and memory

Use the `scalenav` environment from the [installation guide](INSTALL.md),
which includes the training dependencies.

For action alignment with `--micro-batch 1 --deepspeed`, use **48 GB GPU memory
per GPU** and CPU optimizer offload. For a single process, **128 GB of system RAM
is recommended**, including checkpoint saves. Follow the
[DeepSpeed setup](INSTALL.md#deepspeed); `--deepspeed` enables
ZeRO-2 with CPU optimizer offload. Add `--text-device cpu` to keep the frozen
text encoder on CPU.

For full-parameter training without sharding or offload, FP32 parameters,
gradients and Adam states alone occupy about **75 GiB**, before activations
and the text encoder. Budget additional RAM for multiple processes using the
[DeepSpeed memory estimator](https://deepspeed.readthedocs.io/en/latest/memory.html#discussion).
CPU RAM and checkpoint storage must accommodate the full optimizer state.

### Native Wan weights

P0 requires the complete Wan model (**34.20 GB**), including its native
transformer shards. Download it to the base-model directory from the
[checkpoint setup](INSTALL.md#checkpoints):

```bash
python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
    local_dir="checkpoints/Wan2.2-TI2V-5B",
)
PYTHON
```

Existing inference components are reused. The commands below use
this directory as `BASE_MODEL`.

## Data preparation

Follow the [data preparation commands](DATA.md) and set `TRAIN_DATA` to the
resulting directory. The trainer accepts two
prepared caches: instruction-conditioned Internet videos and action-labeled
simulation episodes. Supply the **training split** of each, not evaluation data.
The action metadata must match the metadata used to build its cache.

Video and action samples each use uniform sampling over all cached chunks
in their respective training split. The action-free replay probability chooses
the source collection; it is not an extra loss multiplier.
The bundled annotations, compact video downloads and scene preparation are described in
[Data](DATA.md).

<a id="run-the-four-phases"></a>

## Run pretraining and post-training

Set the prepared data directory and native Wan model:

```bash
export BASE_MODEL=checkpoints/Wan2.2-TI2V-5B
export TRAIN_DATA="${TRAIN_DATA:-data/training/goto}"
export RUN=outputs/scalenav
```

### Video pretraining

```bash
awm train --stage pretrain --gpus 8 \
  --base-model "$BASE_MODEL" --video-cache "$TRAIN_DATA/video-cache/train" \
  --output "$RUN/pretrain"
```

This runs P0 and then P1 in separate worker processes. P1 restores P0's optimizer
and RNG state and continues the sampling sequence. Phase checkpoints are saved
under `$RUN/pretrain/p0/` and `$RUN/pretrain/p1/`. The `final` link points to the
completed P1 checkpoint.

### Action post-training

```bash
awm train --stage posttrain --gpus 8 \
  --base-model "$BASE_MODEL" --video-cache "$TRAIN_DATA/video-cache/train" \
  --action-cache "$TRAIN_DATA/action-cache" \
  --action-meta "$TRAIN_DATA/simulation/awm_meta.jsonl" \
  --action-split "$TRAIN_DATA/simulation/train.txt" \
  --initialize-from "$RUN/pretrain/final" --output "$RUN/posttrain"
```

This runs action alignment and then joint fine-tuning, initializing each phase
from the preceding model. Each phase starts its own optimizer. Phase checkpoints
are saved under `$RUN/posttrain/align/` and `$RUN/posttrain/joint/`; `final` points
to the completed joint policy. `--initialize-from` can also name an existing
completed P1 checkpoint directly.

### Runtime options

| Option | Effect |
| --- | --- |
| `--gpus 8` | Launch eight GPU workers; do not wrap these commands in `torchrun`. |
| `--micro-batch 1` | Samples per GPU; accumulation preserves each phase's configured global batch. |
| `--deepspeed` | Use ZeRO-2 with CPU optimizer offload after installing [DeepSpeed](INSTALL.md#deepspeed). |
| `--text-device cpu` | Keep the frozen text encoder on CPU. |
| `--save-every 1000` | Save intermediate checkpoints for recovery; by default, each phase saves its final checkpoint. |
| `--dry-run` | Print the resolved phase configurations and worker commands without training. |
| `--check-data` | Validate inputs for both phases without loading model weights or optimizing. |
| `--max-updates 1` | Stop the current phase early for a smoke run; later phases wait until it completes. |

Learning rates, phase lengths, global batches and replay ratios come from
[`training.yaml`](../src/awm/configs/training.yaml). The default eight-process
setup uses accumulation 8 for P0/P1/alignment and 4 for joint fine-tuning.
A phase failure stops the pipeline. The `final` link is created only after both
phases complete; checkpoints are never automatically deleted.

## Loss and history

Inputs are 512×288 RGB at 4 Hz. Chunks contain nine observations and nine actions;
the last action transitions to the next chunk's first observation. Actions are
preceding-body-frame `(dx, dy, dyaw)` displacements, normalized with symlog c=0.01.
P0 has no history. Later phases retain the initial-frame anchor and two recent
chunks, with clean history and absolute temporal positions.

Video-only samples have zero action tokens. Action-labeled samples use independently
sampled video/action noise levels. The two flow MSEs use detached inverse-loss
weights normalized to one. In a mixed minibatch, action-free samples contribute
their own video loss, outside the action samples' balancing weights. The losses
are combined by their minibatch sample fractions.

Training draws independent indices from the original 1,000-level shifted video
schedule for both modalities (shift 5). The paper specifies separate shifts 5/1
for **inference**; those are applied only by the inference sampler.
The known current-observation latent is kept clean and excluded from the future
video denoising loss. A terminal minibatch with no valid future video positions
uses its action loss directly. Final action chunks append physical zero actions before normalization and keep
these in action supervision. Repeated terminal video padding is excluded with
the temporally max-pooled frame-validity mask.

## Resume and use trained weights

Resume with the same pipeline, output directory, data, GPU count and micro-batch.
The checkpoint identifies the phase: earlier phases are skipped, the current
phase resumes, and the remaining phase runs automatically. Model, optimizer,
LR scheduler, RNG and sampling position are restored.

For example, after saving intermediate checkpoints with `--save-every 1000`:

```bash
awm train --stage posttrain --gpus 8 \
  --base-model "$BASE_MODEL" --video-cache "$TRAIN_DATA/video-cache/train" \
  --action-cache "$TRAIN_DATA/action-cache" \
  --action-meta "$TRAIN_DATA/simulation/awm_meta.jsonl" \
  --action-split "$TRAIN_DATA/simulation/train.txt" \
  --resume "$RUN/posttrain/joint/checkpoint-1000" \
  --output "$RUN/posttrain" --save-every 1000
```

Keep the original offload and text-device options when resuming. A checkpoint
at a phase boundary also works: its completed phase is checked before continuing
to the next. Resuming a completed `final` checkpoint performs no extra updates.

Each checkpoint contains a portable BF16 transformer export and `state/` with the
training state. Model weights and `train_meta.pt` suffice for inference; training
resume also requires `state/`.

Use the final policy with the [paired evaluation example](INFERENCE.md#paired-evaluation-example):

```bash
awm infer --checkpoint "$RUN/posttrain/final" --base-model "$BASE_MODEL" \
  --image data/example/observation.png --prompt "$(cat data/example/instruction.txt)" --output outputs/prediction
```

Open `outputs/prediction/observation_trajectory.png` to inspect the predicted path.
