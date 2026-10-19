# Training

The supported sequence is P0 → P1 → action alignment → joint fine-tuning.
P0 and P1 together form Stage 1. The recipe is installed with the package in
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

Use the `scalenav` environment from the [README installation](../README.md#installation),
which includes the training dependencies.

For action alignment with `--micro-batch 1 --deepspeed`, use **48 GB GPU memory
per GPU** and CPU optimizer offload. For a single process, **128 GB of system RAM
is recommended**, including checkpoint saves. Follow the README's
[DeepSpeed setup](../README.md#optional-dependencies); `--deepspeed` enables
ZeRO-2 with CPU optimizer offload. Add `--text-device cpu` to keep the frozen
text encoder on CPU.

For full-parameter training without sharding or offload, FP32 parameters,
gradients and Adam states alone occupy about **75 GiB**, before activations
and the text encoder. Budget additional RAM for multiple processes using the
[DeepSpeed memory estimator](https://deepspeed.readthedocs.io/en/latest/memory.html#discussion).
CPU RAM and checkpoint storage must accommodate the full optimizer state.

### Native Wan weights

P0 requires the complete Wan model (**34.20 GB**), including its native
transformer shards. Download it to the same base-model directory used by the README:

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

Existing inference components are reused. The four-phase commands below use
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

## Run the four phases

Set paths to prepared data and native Wan weights:

```bash
export BASE_MODEL=checkpoints/Wan2.2-TI2V-5B
export TRAIN_DATA="${TRAIN_DATA:-data/training/goto}"
export VIDEO_CACHE="$TRAIN_DATA/video-cache/train"
export ACTION_CACHE="$TRAIN_DATA/action-cache"
export ACTION_META="$TRAIN_DATA/simulation/awm_meta.jsonl"
export ACTION_SPLIT="$TRAIN_DATA/simulation/train.txt"
export RUN=outputs/scalenav
```

First validate the video cache without loading the model:

```bash
awm train --stage p0 --base-model "$BASE_MODEL" --video-cache "$VIDEO_CACHE" \
  --output "$RUN/p0" --check-data
```

Run with eight processes and a micro-batch of one. Accumulation is eight for
P0/P1/alignment and four for joint fine-tuning. Use `--deepspeed` on each command
if optimizer offload is needed. The native Wan transformer shards are required
for P0; inference needs only the released transformer plus Wan's VAE/text assets.

```bash
torchrun --standalone --nproc_per_node=8 --module awm.training.runner \
  --stage p0 --base-model "$BASE_MODEL" --video-cache "$VIDEO_CACHE" \
  --output "$RUN/p0" --micro-batch 1 --save-every 2000

torchrun --standalone --nproc_per_node=8 --module awm.training.runner \
  --stage p1 --base-model "$BASE_MODEL" --video-cache "$VIDEO_CACHE" \
  --initialize-from "$RUN/p0/checkpoint-2000" \
  --output "$RUN/p1" --micro-batch 1 --save-every 8000

torchrun --standalone --nproc_per_node=8 --module awm.training.runner \
  --stage align --base-model "$BASE_MODEL" --video-cache "$VIDEO_CACHE" \
  --action-cache "$ACTION_CACHE" --action-meta "$ACTION_META" --action-split "$ACTION_SPLIT" \
  --initialize-from "$RUN/p1/checkpoint-10000" \
  --output "$RUN/align" --micro-batch 1 --save-every 2500

torchrun --standalone --nproc_per_node=8 --module awm.training.runner \
  --stage joint --base-model "$BASE_MODEL" --video-cache "$VIDEO_CACHE" \
  --action-cache "$ACTION_CACHE" --action-meta "$ACTION_META" --action-split "$ACTION_SPLIT" \
  --initialize-from "$RUN/align/checkpoint-2500" \
  --output "$RUN/joint" --micro-batch 1 --save-every 6500
```

These examples retain the final checkpoint of each phase. Choose a smaller
`--save-every` for recovery during long runs, with correspondingly higher storage
use. Checkpoints are never automatically deleted. `--max-updates` stops a smoke
run early without modifying the underlying LR schedule; the configured training schedule remains unchanged. `--dry-run` prints the resolved recipe.

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

`--initialize-from` starts the next stage. P0→P1 carries optimizer state as well;
alignment and joint fine-tuning start new optimizers. `--resume` continues the
same stage and restores model, optimizer, LR scheduler and RNG state. Its recipe,
data fingerprints, seed, world size and micro-batch must match the saved run.
The stateless batch sampler also resumes at the exact remaining sample position.

```bash
awm train --stage joint --base-model "$BASE_MODEL" --video-cache "$VIDEO_CACHE" \
  --action-cache "$ACTION_CACHE" --action-meta "$ACTION_META" --action-split "$ACTION_SPLIT" \
  --resume "$RUN/joint/checkpoint-1000" --output "$RUN/joint"
```

Use the same distributed launcher and micro-batch as the original run. The
single-process example above applies only to a single-process original run.
Weights and `train_meta.pt` alone are sufficient for inference, but cannot
restore an optimizer. Each new checkpoint contains both a portable BF16
transformer export and `state/` with the full training state.

Use the paired image and instruction from
[paired evaluation example](INFERENCE.md#paired-evaluation-example):

```bash
awm infer --checkpoint "$RUN/joint/checkpoint-6500" --base-model "$BASE_MODEL" \
  --image data/example/observation.png --prompt "$(cat data/example/instruction.txt)" --output outputs/prediction
```

Open `outputs/prediction/observation_trajectory.png` to inspect the predicted path.
