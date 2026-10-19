# Simulation evaluation

First complete the README's [installation](../README.md#installation) and
[checkpoint downloads](../README.md#checkpoints). Run the commands below from
the ScaleNav repository root with the `scalenav` environment active.

1. [Install the simulator](#simulator-installation).
2. [Choose and prepare an evaluation set](#evaluation-data).
3. Run [Habitat-GS-LN](#run-habitat-gs-ln-evaluation) or
   [Short-Horizon OVON](#run-short-horizon-ovon-evaluation).
4. [View metrics and videos](#visualizations).

## Simulator installation

Complete the README's [simulator installation](../README.md#simulator-installation),
including the final `awm doctor --require-cuda --data-tools --simulator` check.
Continue below if that check has already passed in your `scalenav` environment.

## Evaluation data

Choose one set. Total data storage includes prepared assets from all sources
and retained downloads, summing file bytes in decimal MB/GB. Model weights,
environments and evaluation outputs are separate.

| Selection | Tasks | Scenes | Total data storage | Prepare |
| --- | ---: | ---: | ---: | --- |
| Habitat-GS-LN test subset | 16 | 13 | **2.96 GB** | [Prepare](#test-subset) |
| Full Habitat-GS-LN | 157 | 45 | **10.36 GB** | [Prepare](#full-habitat-gs-ln) |
| Short-Horizon OVON test subset | 7 | 7 | **0.81 GB** | [Prepare](#short-horizon-ovon) |
| Full Short-Horizon OVON | 524 | 36 | **3.67 GB** | [Prepare](#short-horizon-ovon) |

Start with a test subset to try navigation. These contain selected successful
ScaleNav cases, with paper examples first; use the full sets for benchmark scores.
Both OVON selections download **5.84 GB** from the official sources. Their storage
totals assume the HM3D archive cleanup below; allow **6.49 GB** for the subset or
**9.34 GB** for the full benchmark before cleanup.

Our task definitions and instructions are included in the repository. Preparation
combines them with the scene assets and creates `episodes.json` and
`scenes.scene_dataset_config.json` under `data/evaluation/<set>/`.
Downloaded assets stay in `data/upstream/evaluation/`; keep this directory for evaluation.

### Test subset

Install the [scene converter](../README.md#scene-converter), request
[InteriorGS access](https://huggingface.co/datasets/spatialverse/InteriorGS)
and accept its [terms](https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf),
then run:

```bash
hf auth login
export PATH="$PWD/external/splat-tools/node_modules/.bin:$PATH"
python scripts/prepare_data.py evaluation --subset test_subset \
  --source data/upstream/evaluation --output data/evaluation/test_subset --download
```

This downloads and prepares the 13 scenes, navmeshes and instance labels
(**2.96 GB** total, including retained compressed scenes).
Prepared data: `data/evaluation/test_subset/`.
Continue to [run Habitat-GS-LN evaluation](#run-habitat-gs-ln-evaluation).

### Full Habitat-GS-LN

Requires [InteriorGS access](https://huggingface.co/datasets/spatialverse/InteriorGS)
under its [terms](https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf).
Install the [scene converter](../README.md#scene-converter), then prepare all 157 tasks:

```bash
hf auth login
export PATH="$PWD/external/splat-tools/node_modules/.bin:$PATH"
python scripts/prepare_data.py evaluation --subset full \
  --source data/upstream/evaluation \
  --output data/evaluation/habitatgs_ln --download
```

This downloads and prepares all 45 scenes, navmeshes and instance labels,
reusing any test-subset downloads. Total data storage is **10.36 GB**, including
retained compressed scenes. Evaluation files: `data/evaluation/habitatgs_ln/`.
Continue to [run Habitat-GS-LN evaluation](#run-habitat-gs-ln-evaluation)
with `EVAL_SET=habitatgs_ln`.

<details>
<summary>Habitat-GS-LN data sources and download paths</summary>

The commands above download only the assets required by the selected set and
verify their pinned SHA-256 hashes. Paths below are relative to
`data/upstream/evaluation/`; `<scene>` is the scene ID.

| Asset | Source | Download path |
| --- | --- | --- |
| Prepared scene visuals | [Habitat-GS assets](https://huggingface.co/datasets/RukawaY/gs_scenes) | `train/interior_<scene>/interior_<scene>.gs.ply` |
| Compressed scenes for the remaining visuals | [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) | `<scene>/3dgs_compressed.ply` |
| Original Habitat-GS navmeshes | [ScaleNav Data](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data/tree/main/evaluation/habitatgs_ln/navmeshes) | `evaluation/habitatgs_ln/navmeshes/<scene>.navmesh` |
| Instance labels | [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) | `<scene>/labels.json` |

The full set downloads six prepared visuals (1.37 GB), 39 compressed scenes
(1.78 GB), 45 instance-label files (24.09 MB) and 45 navmeshes (980,920 bytes).
The converter writes the remaining visuals to `scenes/<scene>/<scene>.gs.ply`.
Target footprints are computed locally from instance labels. Original navmeshes
are distributed with the Habitat-GS authors' permission.
Task definitions and asset checksums are in
[the evaluation annotations](../annotations/evaluation/habitatgs_ln.json).

</details>

### Short-Horizon OVON

**1. Download the HM3D validation assets.** Follow the
[official access instructions](https://github.com/matterport/habitat-matterport-3dresearch),
then download these two v0.2 archives:

- [hm3d-val-habitat-v0.2.tar](https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar)
- [hm3d-val-semantic-annots-v0.2.tar](https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar)

**2. Prepare a selection.** Set `HM3D_ARCHIVES` to their download directory.
The command extracts only the required scene visuals, navmeshes and semantic
labels, verifies their SHA-256 hashes, downloads the official task metadata and
applies our instructions. The scene configuration is generated locally.

For the 7 selected test cases:

```bash
export HM3D_ARCHIVES=/path/to/hm3d-archives
python scripts/prepare_data.py ovon --subset test_subset \
  --hm3d-archives "$HM3D_ARCHIVES" \
  --source data/upstream/evaluation/ovon \
  --scenes data/upstream/evaluation/hm3d \
  --output data/evaluation/ovon_test_subset --download
```

For the full 524-task benchmark, use the same archives:

```bash
export HM3D_ARCHIVES=/path/to/hm3d-archives
python scripts/prepare_data.py ovon --subset full \
  --hm3d-archives "$HM3D_ARCHIVES" \
  --source data/upstream/evaluation/ovon \
  --scenes data/upstream/evaluation/hm3d \
  --output data/evaluation/short_horizon_ovon --download
```

Existing matching files are reused. If the scenes are already extracted,
omit `--hm3d-archives`. Existing unselected scenes are left untouched.

**3. Remove source archives after successful preparation.** Evaluation uses the
extracted files, so these two archives are no longer needed:

```bash
rm -- "$HM3D_ARCHIVES/hm3d-val-habitat-v0.2.tar" \
  "$HM3D_ARCHIVES/hm3d-val-semantic-annots-v0.2.tar"
```

Keep the archives if you plan to prepare the full benchmark later; this adds
**5.68 GB**. Keep `data/upstream/evaluation/hm3d/` and
`data/upstream/evaluation/ovon/` for evaluation and re-preparation.

| Selection | Required scenes | Storage after archive cleanup | Storage with archives |
| --- | ---: | ---: | ---: |
| Test subset | 7 (641.72 MB) | **0.81 GB** | **6.49 GB** |
| Full benchmark | 36 (3.48 GB) | **3.67 GB** | **9.34 GB** |

These totals are for a fresh preparation and include the retained OVON task
archive and prepared metadata. If you previously extracted all HM3D validation
scenes, those extra files still occupy space.
Prepared data: `data/evaluation/ovon_test_subset/` or
`data/evaluation/short_horizon_ovon/`.
Continue to [run Short-Horizon OVON evaluation](#run-short-horizon-ovon-evaluation).

<details>
<summary>OVON download sizes and sources</summary>

| File | Bytes |
| --- | ---: |
| `hm3d-val-habitat-v0.2.tar` | 3,530,618,880 |
| `hm3d-val-semantic-annots-v0.2.tar` | 2,145,935,360 |
| [ABot starting poses](https://huggingface.co/datasets/acvlab/ABotN-Short-Horizon-OVON) | 1,016,534 |
| [HM3D-OVON annotations](https://huggingface.co/datasets/nyokoyama/hm3d_ovon) | 165,665,450 |
| **Download payload** | **5,843,236,224 (5.84 GB)** |

Both selections download the same official archives. Selective extraction reduces
local storage, not the initial download. Task definitions are in
[the OVON annotations](../annotations/evaluation/short_horizon_ovon.json).

</details>

## Run Habitat-GS-LN evaluation

Choose the prepared set: `test_subset` for 16 tasks or `habitatgs_ln` for all 157.
Set `CHECKPOINT` and `BASE_MODEL` using the [README](../README.md#checkpoints).

```bash
export EVAL_SET=test_subset
python -m awm.scripts.eval_habitat_gs \
  --eval_config src/awm/configs/eval_habitatgs.yaml \
  --episode_manifest data/evaluation/"$EVAL_SET"/episodes.json \
  --gs_dataset_root data/upstream/evaluation \
  --scene_dataset_config data/evaluation/"$EVAL_SET"/scenes.scene_dataset_config.json \
  --model_path "$CHECKPOINT" --pretrained_model_path "$BASE_MODEL" \
  --visualize --output_dir outputs/evaluation/"$EVAL_SET"
```

## Run Short-Horizon OVON evaluation

Choose `ovon_test_subset` for the 7 selected test cases, or
`short_horizon_ovon` for all 524 tasks. Use the same `CHECKPOINT` and `BASE_MODEL`:

```bash
export EVAL_SET=ovon_test_subset
python -m awm.scripts.eval_habitat_gs \
  --eval_config src/awm/configs/eval_ovon.yaml \
  --episode_manifest data/evaluation/"$EVAL_SET"/episodes.json \
  --gs_dataset_root data/evaluation/"$EVAL_SET" \
  --scene_dataset_config data/evaluation/"$EVAL_SET"/scenes.scene_dataset_config.json \
  --model_path "$CHECKPOINT" --pretrained_model_path "$BASE_MODEL" \
  --visualize --output_dir outputs/evaluation/"$EVAL_SET"
```

## Visualizations

Both evaluation commands save results to `outputs/evaluation/<set>/`, where
`<set>` is `test_subset`, `habitatgs_ln`, `ovon_test_subset` or `short_horizon_ovon`:

| File | What to do |
| --- | --- |
| `summary.json` | Read `aggregate.success_rate` (SR), `aggregate.spl` (SPL) and per-task `episodes`. |
| `gallery/index.html` | Open in a browser to view each case's video, trajectory, target and outcome. |
| `videos/` | Find the rollout videos. |

`--visualize` includes every case, including failures. Omit it for metrics only.

<details>
<summary>Build a gallery from saved rollouts</summary>

For example, for the test subset:

```bash
python -m awm.scripts.visualize_evaluation \
  --summary outputs/evaluation/test_subset/summary.json \
  --episodes data/evaluation/test_subset/episodes.json \
  --output outputs/evaluation/test_subset/gallery
```

</details>

## Evaluation protocol

The provided configurations set the evaluation protocol automatically.

<details>
<summary>Success, STOP and SPL definitions</summary>

The evaluator uses 200 actions,
model-triggered STOP, and capsule-to-target-footprint distance at most 1.25 m.
Short-Horizon OVON uses the strict threshold < 1.25 m.
STOP requires nine consecutive issued actions with planar norm < 0.015 m and
absolute yaw < 0.015 rad, and at least 18 issued actions in the episode. The capsule has radius 0.25 m and centerline length 0.36 m;
camera HFOV is 69.62°. Facing direction does not gate success. SPL uses navmesh geodesic distance;
missing scenes, required footprints, or navigable reference paths raise an error.
Habitat-GS-LN's three doorway tasks use their annotated navigation endpoint
(geodesic distance < 1.25 m and model STOP); object targets use the oriented footprint.

</details>

## Episode format

Preparation generates this format automatically. Refer to it when adding custom tasks.

<details>
<summary>Manifest fields and target definitions</summary>

A manifest is a JSON object with `benchmark` (`habitatgs_ln` or
`short_horizon_ovon`) and an `episodes` list. Each episode requires:

- `scene_id`, `episode_id`, `instruction`: nonempty strings.
- `start_xyz`, `goal_xyz`: world positions in metres; `start_quat_wxyz`: starting
  orientation. The Habitat-GS reference goal is its annotated `goal_xyz`.
- `action_convention`: `scan2d_body_forward_right_clockwise`.
- `scene_path`, `navmesh_path`: asset paths, absolute or relative to
  `--gs_dataset_root`. Habitat-GS can also use the existing scene-ID layout.
- `target_contract`: `semantic_target_footprint_world_xz` for a single object target,
  or `supporting_instances`, each with `obj_id` and `footprint_world_xz`.
  Footprints are polygons with at least three world-XZ vertices.
- The three Habitat-GS-LN doorway tasks use `target_mode: doorway` and
  `target_contract.success_condition.type: distance_to_navigation_goal`.
- Camera parameters: `cam_height_m`, `cam_pitch_deg`, `cam_forward_m`, `hfov_deg`.
  In prepared manifests, `start_xyz` includes camera height; the forward offset
  is applied by the sensor without shifting the robot's navigation position.
- For Short-Horizon OVON, `reference_goal_positions`: navigable world-XYZ
  points sampled near all valid target footprints. SPL uses the minimum
  navmesh distance to these reference points.

Habitat-GS-LN specifies one target instance or a doorway endpoint per episode.
Short-Horizon OVON has 524 episodes in 36 scenes and permits all valid instances
of the requested category. Front-visibility filtering selects the fixed episodes;
it does not restrict success to only the initially visible target instance.
The prepared OVON contract filters category instances by the robot's final floor,
falling back to all category instances when no same-floor instance exists.
SPL reference points use the starting floor and navmesh island. For multiple
instances, use `supporting_instances` without a direct single-target footprint
or a restrictive `success_condition.support_obj_ids` list.

</details>
