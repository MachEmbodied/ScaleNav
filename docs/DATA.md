# Data

## Choose a download

[ScaleNav Data](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data#choose-a-dataset)
is the shared entry point for all selections below. Each dataset name links to
its complete download and preparation instructions.

Evaluation and training use separate preparation commands and directories. To try
navigation, start with the **[16-case Habitat-GS-LN test subset](EVALUATION.md#test-subset)**;
it does not download any training data. **Network download** is the amount
transferred over the Internet from all sources. **Disk usage after preparation**
is the local space occupied by the prepared data and retained downloads after
completing the guide, including any cleanup.
Sizes sum file bytes in decimal units and count each selection separately;
shared files are reused across selections. Model weights, environments,
training latent caches and run outputs are separate.

| Dataset | Purpose | Contents | Network download | Disk usage after preparation |
| --- | --- | --- | ---: | ---: |
| [Habitat-GS-LN test subset](EVALUATION.md#test-subset) | Preview navigation and visualizations | 16 tasks / 13 scenes | 991.32 MB | 2.96 GB |
| [Full Habitat-GS-LN evaluation](EVALUATION.md#full-habitat-gs-ln) | Report Habitat-GS-LN SR/SPL | 157 tasks / 45 scenes | 3.17 GB | 10.36 GB |
| [Short-Horizon OVON test subset](EVALUATION.md#ovon-test-subset) | Preview navigation on HM3D | 7 tasks / 7 scenes | 642.85 MB | 642.90 MB |
| [Full Short-Horizon OVON evaluation](EVALUATION.md#full-short-horizon-ovon) | Report Short-Horizon OVON SR/SPL | 524 tasks / 36 scenes | 5.84 GB | 3.67 GB |
| [Small training subset (for preview)](#small-training-subset) | Check the training pipeline | 17 Internet segments / 4 simulated episodes | 55.60 MB | 239.54 MB |
| [Full training data](#full-training-data) | Pretrain and fine-tune ScaleNav | 18,249 Internet segments / 2,600 simulated episodes, including validation | 36.10 GB | 153.04 GB |

Totals follow the documented preparation steps. Habitat-GS-LN retains compressed
scenes alongside converted visuals, labels and navmeshes. OVON extracts only the
selected scenes. The [OVON test subset](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data#ovon-test-subset)
downloads individual files after a one-time Matterport agreement confirmation.
Full OVON removes the two HM3D source archives after preparation. Keeping those
archives adds **5.68 GB**; allow **9.34 GB** before full-OVON cleanup. Both training totals include
converted scenes, locally rendered videos, generated labels and source downloads.

<details>
<summary>Exact network download sizes</summary>

The network-download column sums these file sizes. An empty cache
requires **991,324,013 bytes** for the Habitat-GS-LN test subset,
**3,173,478,248 bytes** for full Habitat-GS-LN, **642,847,267 bytes** for the
OVON test subset, **5,843,236,224 bytes** for full OVON, **55,603,886 bytes** for the small training subset and
**36,103,149,859 bytes** for full training. The shared video index is counted
once. Existing matching files reduce the download.

</details>

The included training labels and recorded poses total only **4.39 MB**;
[evaluation annotations](../annotations/evaluation/) add **316.57 KB** for both benchmarks.
Prepared evaluation files are stored under `data/evaluation/`. Follow the
[full evaluation guide](EVALUATION.md#evaluation-data) for benchmark scores or
[test-subset instructions](EVALUATION.md#test-subsets) for navigation previews.
Internet validation checks held-out video examples; it is separate from the two
closed-loop navigation benchmarks. Full Internet training and validation contain
8.63 GB of MP4 files, counted alongside the retained TAR packs in the full
training total. `--limit` limits prepared samples, not archive size; use `--subset smoke`
for the independently downloadable small training pack.

## Small training subset

Preview the training data with **17 Internet segments and 4 simulated episodes**,
including their ScaleNav instructions, splits, trajectories and actions
(**239.54 MB** total data storage after preparation).
The annotations are included in this repository; the commands below download
the matching Internet videos and one official scene, render the four episodes,
and write the paired videos and labels together.

The Internet portion contains 17 instruction-labeled samples from 14 videos
(16 distinct frame intervals), each 2.25–6.75 seconds at 512×288 / 4 Hz.
Two samples pair the same interval with different instructions. Targets include
sofas, sinks, drawers, a fireplace, plants, wine storage and a foosball table.
The four simulated episodes share scene `0003_839989`:

| Episode | Target | Frames | Video duration | Action steps |
| --- | --- | ---: | ---: | ---: |
| 1 | Sofa | 62 | 15.50 s | 61 |
| 2 | Sofa ahead | 48 | 12.00 s | 47 |
| 3 | Table ahead | 26 | 6.50 s | 25 |
| 4 | Area near the plant | 34 | 8.50 s | 33 |

The download is **55,603,886 bytes**: **10,588,160 bytes** for the Internet video
pack, **1,072,212 bytes** for its shared index and **43,943,514 bytes** for the
official compressed scene. Simulation videos are rendered locally from the
included poses. This subset checks data preparation, latent caching and training;
all its samples belong to the training split.

Complete the [installation guide](INSTALL.md), including the
simulator and scene converter,
then obtain [InteriorGS access](https://huggingface.co/datasets/spatialverse/InteriorGS)
under its [terms](#upstream-access-and-terms). From the repository root:

```bash
hf auth login
export PATH="$PWD/external/splat-tools/node_modules/.bin:$PATH"
export TRAIN_DATA=data/smoke/goto
python scripts/prepare_data.py internet --subset smoke --split train \
  --source data/downloads/scalenav --output "$TRAIN_DATA"/internet \
  --instructions goto --download
python scripts/prepare_data.py simulation --subset smoke --split train \
  --source data/upstream/interiorgs --scenes data/scenes \
  --output "$TRAIN_DATA"/simulation --instructions goto --download --gpu 0
```

The complete prepared data is under `data/smoke/goto/internet/` and
`data/smoke/goto/simulation/`. For training, continue with
[Build training caches](#build-training-caches) using the same `TRAIN_DATA`.
All selected examples belong to the training split; use the full collection
for the complete training recipe.

## Full training data

Prepare **18,249 Internet segments and 2,600 simulated episodes**, including
training and validation splits, with their ScaleNav instructions, trajectories
and actions. Total data storage after preparation is **153.04 GB**, including
all downloaded sources, converted scenes, videos and generated training labels.
Our annotations are included in this repository and applied automatically.

<details>
<summary>Full training storage breakdown</summary>

| Component | Storage |
| --- | ---: |
| Retained Internet video packs and index | 8.64 GB |
| Retained InteriorGS compressed scenes | 27.46 GB |
| 650 converted scenes | 105.75 GB |
| Prepared Internet videos and labels | 8.63 GB |
| 2,600 rendered simulation videos, actions and trajectories | 2.56 GB |
| **Total** | **153.04 GB** |

</details>

Complete the [installation guide](INSTALL.md), including the
simulator and scene converter,
then obtain [InteriorGS access](https://huggingface.co/datasets/spatialverse/InteriorGS)
under its [terms](#upstream-access-and-terms). From the repository root:

```bash
hf auth login
export PATH="$PWD/external/splat-tools/node_modules/.bin:$PATH"
export TRAIN_DATA=data/training/goto
python scripts/prepare_data.py internet --subset full --split all \
  --source data/downloads/scalenav --output "$TRAIN_DATA"/internet \
  --instructions goto --download
python scripts/prepare_data.py simulation --subset full --split all \
  --source data/upstream/interiorgs --scenes data/scenes \
  --output "$TRAIN_DATA"/simulation --instructions goto --download --gpu 0
```

The complete prepared data is under `data/training/goto/internet/` and
`data/training/goto/simulation/`. For training, continue with
[Build training caches](#build-training-caches) using the same `TRAIN_DATA`.
The small subset and full collection share the download and scene caches;
existing matching files are reused.

## Training preparation environment

All dependencies are installed in the [installation guide](INSTALL.md),
including [Habitat-GS](INSTALL.md#simulator-installation) for rendering and the
[scene converter](INSTALL.md#scene-converter). Run data preparation from the
repository root with `scalenav` active.

## Build training caches

After preparing either training selection, keep its `TRAIN_DATA` setting and
use the checkpoint/base-model paths from the [checkpoint setup](INSTALL.md#checkpoints):

```bash
python -m awm.scripts.precompute_nav_t2v_latents \
  --config "$TRAIN_DATA"/internet/train.yaml \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir "$TRAIN_DATA"/video-cache/train --encode_batch_size 4
python -m awm.scripts.precompute_nav_wam_latents \
  --meta_path "$TRAIN_DATA"/simulation/awm_meta.jsonl \
  --data_root "$TRAIN_DATA"/simulation \
  --pretrained_model_path "$BASE_MODEL" \
  --output_dir "$TRAIN_DATA"/action-cache --encode_batch_size 4

awm train --stage joint --base-model "$BASE_MODEL" --initialize-from "$CHECKPOINT" \
  --video-cache "$TRAIN_DATA"/video-cache/train \
  --action-cache "$TRAIN_DATA"/action-cache \
  --action-meta "$TRAIN_DATA"/simulation/awm_meta.jsonl \
  --action-split "$TRAIN_DATA"/simulation/train.txt \
  --output outputs/data-check --check-data --workers 0
```

`--check-data` checks the prepared caches, metadata and split without running
optimization. Continue with the [training commands](TRAINING.md) using these paths.

## Included annotations

[annotations/](../annotations/) contains our navigation labels and fixed splits:

| Collection | Training | Validation | Included fields |
| --- | ---: | ---: | --- |
| Internet video | 18,184 segments | 65 segments | 13,647 upstream clip IDs, archive locations, frame boundaries, instructions |
| Simulation | 2,460 episodes | 140 episodes | 650 scene IDs, instructions, 4 Hz poses, actions, camera parameters |

Internet splits are disjoint by SpatialVID clip ID; simulation splits are disjoint
by scene (618 training / 32 validation). `manifest.json` pins upstream revisions
and SHA-256 checksums. `simulation.npz` stores packed float32 pose/action arrays;
`simulation.jsonl.gz` supplies their offsets. No pickle is needed.

Annotations and released Internet videos use
[CC BY-NC-SA 4.0](../annotations/LICENSE). Code uses Apache-2.0.
[ScaleNav Data](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data) supplies
the selected SpatialVID-derived videos at 512×288 / 4 Hz, with no additional
video conversion required. The preparation script downloads only the packs
needed by the selected split or small training subset and verifies SHA-256.
It does not fetch original SpatialVID archives unless `--from-upstream` is set.

The download guides combine these annotations with the required videos or
scene assets. No separate annotation download or manual pairing is needed.

## Construction and annotation prompts

The following templates are the prompts in the paper's data-construction
appendix. The paper uses Gemini 3.5 Flash for annotation and paraphrasing.

| Step | Inputs and acceptance rule | Prompt |
| --- | --- | --- |
| Internet segment extraction | Uniformly sampled keyframes, in order, labeled with source-frame indices. Identify an arrival endpoint, then search backward for the earliest keyframe showing the same target; generate an initial `Go to ...` instruction. | [Extraction](../annotations/prompts/internet_segment_extraction.txt) |
| Segment review | Instruction, target description, segment frames, and **separately labeled source frames around the endpoint**. Retain a segment only if target–instruction consistency, motion plausibility, arrival accuracy, and temporal continuity all pass. An uncertain judgment is not a pass. | [Review](../annotations/prompts/internet_segment_review.txt) |
| Simulated instruction refinement | Initial observation, selected target and scene object annotations. Resolve ambiguity using visible attributes or spatial relations without changing the target; discard unresolved references. | [Refinement](../annotations/prompts/simulated_instruction_refinement.txt) |
| Simulated segment review | Apply the four review criteria with scene annotations and a target-marked reference image. The marker identifies the target; it is not an object attribute or instruction content. | [Review addendum](../annotations/prompts/simulated_review_addendum.txt) |
| Accepted-instruction paraphrasing | Paraphrase only accepted instructions, preserving the target, destination, distinguishing attributes and directional references. | [Paraphrasing](../annotations/prompts/instruction_paraphrasing.txt) |

For Internet videos, SpatialVID-HQ scene annotations select approximately 60K
indoor candidate clips. The extraction stage maps source-frame indices to
timestamps and retains a **continuous source-video segment**, not a slideshow
of the sampled keyframes. Filtering yields 18,249 accepted segments. The released
`t0` and `t_end` are already mapped to the processed **4 Hz** video timeline.
Use these indices directly after resampling, with the interval `[t0, t_end)`.

For simulation, the candidate pool contains approximately 1K InteriorGS scenes.
After scene-quality filtering, starting poses are sampled from traversable
navmesh areas and targets are selected from the initial observations. An offline
2D adaptation of SCAN-Planner uses A* guidance, footprint-aware optimization,
time reparameterization and collision checking. Poses and rendered observations
are sampled at 4 Hz; consecutive poses yield preceding-body-frame action
increments. Review retains 2,600 episodes from 650 scenes. Both collections then
undergo the accepted-instruction paraphrasing step described in the paper.

The preparation commands reconstruct training inputs from the included
accepted labels, fixed splits and recorded trajectories. They do not repeat
VLM annotation, resample targets or rerun the planner. This preserves the released
sample identities and action labels. The prompts specify how to annotate new
candidates; they are not policy inference prompts. At inference the policy takes
the navigation instruction directly, with no VLM review or chat template.

## Upstream access and terms

1. Internet video packs and ScaleNav annotations are publicly downloadable.
   For simulation data, request [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS)
   access on its official page, review its terms, then authenticate with
   `hf auth login`. Preparation downloads only the selected scene files;
   the small training subset uses one scene. SpatialVID upstream access is needed only for the optional
   [original-archive reconstruction](#rebuild-from-original-spatialvid-archives).
2. SpatialVID data and our compact video packs use **CC BY-NC-SA 4.0**: attribute its authors, use it
   noncommercially, identify changes and apply the same license to adaptations.
   Its [code license](https://github.com/NJU-3DV/SpatialVID#license) is separate.
   Each video pack includes the full license, a
   [source/modification notice](../annotations/SPATIALVID_NOTICE.txt), and
   annotations mapping clips to their upstream archive and source-video paths.
   These terms grant no additional rights to the underlying third-party footage.
3. The [InteriorGS terms](https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf)
   restrict use to noncommercial research/education and prohibit redistributing
   downloaded data, in whole or part. Download scenes from upstream; render your
   training videos locally. Cite the [InteriorGS project](https://github.com/manycore-research/InteriorGS)
   in publications using it. The prepared Habitat-GS example also references
   InteriorGS and retains these underlying restrictions.

4. The [OVON test subset](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data#ovon-test-subset)
   uses a separate file repository with automatic access after accepting the
   [Matterport agreement](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data).
   Its [preparation command](EVALUATION.md#ovon-test-subset) downloads the selected
   HM3D scene and task files plus official ABotN starting poses. These assets
   retain their original terms. Full OVON uses the official HM3D archives.

InteriorGS scenes come from the upstream sources listed in each guide.
Simulation videos and training latent caches are generated locally.

## Prepare Internet video

The script uses the fixed Hugging Face revision and hashes in
[`internet-media.json`](../annotations/internet-media.json). It verifies the
index, each downloaded pack and each extracted video, then writes `videos/`,
`train/`, `val/`, `train.yaml` and `val.yaml`. Video bytes and zero-based,
end-exclusive annotation intervals are preserved. Valid existing files are reused.
Remove `--download` to use packs already present in `--source` without network
access. Use `--split train` for training videos only, or `--split val` and a
separate output directory for the 37.63 MB validation download, including its index.

### Rebuild from original SpatialVID archives

To reconstruct videos from original source material, obtain
[SpatialVID-HQ access](https://huggingface.co/datasets/FelixYuan-YF/SpatialVID-HQ)
and explicitly pass `--from-upstream`. This optional path can download **1.07 TB**
across 74 archives; it is not needed for the compact release above.

```bash
python scripts/prepare_data.py internet --from-upstream \
  --source data/upstream/spatialvid --output data/rebuilt/goto/internet \
  --instructions goto --download
```

The pinned original revision is recorded in `annotations/manifest.json`.
The script extracts selected clips, resamples to 4 Hz / 512×288, and preserves
terminal observations with FFmpeg `eof_action=pass`. It removes temporary
extracted clips after conversion and retains downloaded archives for reuse.
Existing official archives can be supplied without `--download`.

## Prepare simulation video

With `--download`, only the 650 referenced scenes are fetched from the official
`spatialverse/InteriorGS` repository. The converter expands the official
compressed Gaussians and changes axes from InteriorGS `(X,Y,Z)` to Habitat
`(X,Z,-Y)`. The script replays our exact recorded poses with per-episode HFOV and
camera forward offset, producing 512×288 / 4 Hz RGB, `actions.npy`, trajectories,
`awm_meta.jsonl`, `train.txt` and `val.txt`. Poses already include camera height;
actions remain relative to the robot body. Navmeshes are unnecessary for replay
of recorded training poses. Scene assets and rendered videos are both counted
in the storage totals above. The [small training subset](#small-training-subset)
needs only one scene. Existing converted Y-up scenes may be supplied as
`--scenes/<scene>/<scene>.gs.ply`.

`diverse_iv` contains paraphrases with explicit initial-view references in both
collections; `diverse` is also available for simulation, and `goto` contains the
initial instructions. Use a separate data directory for each instruction set (for example,
`data/training/diverse_iv` with `--instructions diverse_iv`).
The paper's paraphrasing prompt preserves directional references; it does not
require appending an initial-view phrase.

For navigation tests, use the [16-case Habitat-GS-LN test subset or full evaluation sets](EVALUATION.md#evaluation-data).

## Input conventions

Prepare RGB videos sampled at **4 Hz**, with instructions referring to a target
visible in the initial frame. The cache scripts resize/crop to 288×512 and
normalize RGB to [-1, 1]. They do not convert arbitrary source frame rates.
Use disjoint training and validation scenes and retain your split manifests.

## Action-free video

One JSONL row describes one extracted navigation segment:

```json
{"video": "videos/segment_00001.mp4", "n_frames": 45, "instruction": "Approach the chair on the left."}
```

Paths are relative to `data_root`; `n_frames` is the number of frames in the
already extracted 4 Hz segment. A segment needs at least nine frames.
Create a YAML file such as:

```yaml
vae_kwargs:
  vae_subpath: Wan2.2_VAE.pth
datasets:
  - type: nav_t2v
    meta_path: video_segments.jsonl
    domains:
      - meta_jsonl: /path/to/video_segments_train.jsonl
        data_root: /path/to/videos-root
```

```bash
python -m awm.scripts.precompute_nav_t2v_latents \
  --config /path/to/video_data.yaml --pretrained_model_path /path/to/Wan2.2-TI2V-5B \
  --output_dir /path/to/video-cache/train --encode_batch_size 4
```

The source collection can also use the original per-clip annotations:
`domains: [{perclip_dir: /path/to/annotations, pool_root: /path/to/pool}]`, where
`pool/videos/<clip>.mp4` is paired with a JSON array of objects containing
`clip`, `t0`, `t_end`, `instruction`, and `confidence`. Boundaries are frame
indices at the supplied video's sampling rate, with `t_end` exclusive.

## Action-labeled episodes

`awm_meta.jsonl` contains one row per episode:

```json
{"key":"scene1/episode1","scene":"scene1","episode":"episode1","video":"scene1/episode1/video.mp4","actions":"scene1/episode1/actions.npy","canonical":"Approach the chair on the left."}
```

`actions.npy` is float32 `[N-1, 3]` for an N-frame video. Every row is the
transition from frame i to i+1 in the preceding robot body's coordinates.
For ScaleNav, the channel convention is **forward, right,
clockwise yaw**, in metres/metres/radians (`scan2d_body_forward_right_clockwise`).
Use physical displacements, not normalized values or velocities. The Habitat and
Go2 adapters convert these raw channels to forward/left/counterclockwise using
`(dx, -dy, -dyaw)`; the Go2 client additionally divides by the action duration.

Alternatively, provide `trajectory` pointing to an NPZ with `position: [N,3]`
and `quat_wxyz: [N,4]`. The conversion uses the source trajectory convention;
explicit `action_delta_body: [N-1,3]` in that NPZ takes precedence. See the
conversion implementation before substituting another simulator's pose axes.
Providing `actions.npy` avoids ambiguity about source pose conventions.

`train.txt` and `val.txt` contain episode keys, one per line. Their contents
must be disjoint. The trainer explicitly requires `--action-split`.

```bash
python -m awm.scripts.precompute_nav_wam_latents \
  --meta_path /path/to/actions/awm_meta.jsonl --data_root /path/to/actions \
  --pretrained_model_path /path/to/Wan2.2-TI2V-5B \
  --output_dir /path/to/action-cache --encode_batch_size 4
```

Both commands also support `torchrun --standalone --nproc_per_node=N --module ...`.
Use the same VAE checkpoint when preparing either cache and running inference.
The scripts store complete video latents and separately encoded first-observation
latents. Do not substitute a slice from a differently encoded cache.

## Cache layout and checks

Video caches contain `manifest.json` and `samples/<sample>/chunk_0000.pt`.
Action caches contain `manifest.json` and `<scene>/<episode>/chunk_0000.pt`.
Each manifest records geometry, source entries and chunk counts. Action cache
metadata also records the source metadata SHA-256. The trainer checks coverage,
geometry and source binding before loading the model, and fails on unreadable
sample files instead of silently replacing the sample.

New cache specifications bind the selected source records, VAE weight SHA-256,
encoder configuration and encoding precision. Action caches also bind the default
camera pitch. Preparation rejects changes to these inputs when resuming; it does
not hash every raw video or action array, so keep raw inputs immutable.

An interrupted cache can be resumed with the same command. A cache is usable
only once all selected samples have been encoded and `complete` is true.
Use a new output directory when changing source data, geometry or VAE weights.
Caches prepared before this fingerprint format remain readable for training,
but must use a new directory if their preparation needs to be resumed.
For a small preparation check use `--max_samples` for videos or `--max_episodes`
for actions; such a subset is not the full training collection.
