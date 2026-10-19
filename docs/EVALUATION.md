# Simulation evaluation

This guide evaluates the **full Habitat-GS-LN (157 tasks)** and
**full Short-Horizon OVON (524 tasks)** benchmarks. Complete the
[installation guide](INSTALL.md) and [checkpoint downloads](INSTALL.md#checkpoints).
Run commands from the repository root with the `scalenav` environment active.

1. Prepare and run [full Habitat-GS-LN](#full-habitat-gs-ln).
2. Prepare and run [full Short-Horizon OVON](#full-short-horizon-ovon).
3. Read [metrics and visualizations](#visualizations).

## Simulator installation

Complete the [simulator installation steps](INSTALL.md#simulator-installation),
including the final `awm doctor --require-cuda --data-tools --simulator` check.
Continue below if that check has already passed in your `scalenav` environment.

## Evaluation data

Each benchmark has its own preparation and evaluation commands below.
**Network download** includes files from all sources. **Disk usage after preparation**
includes prepared assets and retained downloads after the documented cleanup.
Sizes use decimal MB/GB; model weights, environments and evaluation outputs are separate.

| Dataset | Contents | Network download | Disk usage after preparation | Instructions |
| --- | --- | ---: | ---: | --- |
| Full Habitat-GS-LN evaluation | 157 tasks / 45 scenes | 3.17 GB | 10.36 GB | [Prepare and run](#full-habitat-gs-ln) |
| Full Short-Horizon OVON evaluation | 524 tasks / 36 scenes | 5.84 GB | 3.67 GB | [Prepare and run](#full-short-horizon-ovon) |

Full OVON needs **9.34 GB** before archive cleanup. Existing matching scene files
are reused. Our task annotations are included in the repository; preparation
combines them with scene assets and creates `episodes.json` and
`scenes.scene_dataset_config.json` under `data/evaluation/<set>/`.
Keep the downloaded assets under `data/upstream/evaluation/` for evaluation.

## Full Habitat-GS-LN

Requires [InteriorGS access](https://huggingface.co/datasets/spatialverse/InteriorGS)
under its [terms](https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf).
Install the [scene converter](INSTALL.md#scene-converter), then prepare all 157 tasks:

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

### Run Habitat-GS-LN evaluation

Run all 157 tasks in the prepared `habitatgs_ln` set.
Set `CHECKPOINT` and `BASE_MODEL` using the [checkpoint setup](INSTALL.md#checkpoints).

```bash
export EVAL_SET=habitatgs_ln
python -m awm.scripts.eval_habitat_gs \
  --eval_config src/awm/configs/eval_habitatgs.yaml \
  --episode_manifest data/evaluation/"$EVAL_SET"/episodes.json \
  --gs_dataset_root data/upstream/evaluation \
  --scene_dataset_config data/evaluation/"$EVAL_SET"/scenes.scene_dataset_config.json \
  --model_path "$CHECKPOINT" --pretrained_model_path "$BASE_MODEL" \
  --visualize --output_dir outputs/evaluation/"$EVAL_SET"
```

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

<a id="short-horizon-ovon"></a>

## Full Short-Horizon OVON

**1. Download the HM3D validation assets.** Follow the
[official access instructions](https://github.com/matterport/habitat-matterport-3dresearch),
then download these two v0.2 archives:

- [hm3d-val-habitat-v0.2.tar](https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar)
- [hm3d-val-semantic-annots-v0.2.tar](https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar)

**2. Prepare all 524 tasks.** Set `HM3D_ARCHIVES` to their download directory:

```bash
export HM3D_ARCHIVES=/path/to/hm3d-archives
python scripts/prepare_data.py ovon --subset full \
  --hm3d-archives "$HM3D_ARCHIVES" \
  --source data/upstream/evaluation/ovon \
  --scenes data/upstream/evaluation/hm3d \
  --output data/evaluation/short_horizon_ovon --download
```

The command extracts only the 36 required scenes, verifies their SHA-256 hashes,
downloads official task metadata and applies our instructions. Previously prepared
test-subset assets are reused. If all scenes are already extracted, omit
`--hm3d-archives`. Existing unselected scenes are left untouched.

**3. Remove source archives after successful preparation.** Evaluation uses the
extracted files, so these two archives are no longer needed:

```bash
rm -- "$HM3D_ARCHIVES/hm3d-val-habitat-v0.2.tar" \
  "$HM3D_ARCHIVES/hm3d-val-semantic-annots-v0.2.tar"
```

Full-OVON storage is **3.67 GB** after cleanup or **9.34 GB** with the archives.
Keep `data/upstream/evaluation/hm3d/` and `data/upstream/evaluation/ovon/`.

### Run Short-Horizon OVON evaluation

Run all 524 tasks in the prepared `short_horizon_ovon` set, using the same
`CHECKPOINT` and `BASE_MODEL`:

```bash
export EVAL_SET=short_horizon_ovon
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
`<set>` is `habitatgs_ln` or `short_horizon_ovon`:

| File | What to do |
| --- | --- |
| `summary.json` | Read `aggregate.success_rate` (SR), `aggregate.spl` (SPL) and per-task `episodes`. |
| `gallery/index.html` | Open in a browser to view each case's video, trajectory, target and outcome. |
| `videos/` | Find the rollout videos. |

`--visualize` includes every case, including failures. Omit it for metrics only.

<details>
<summary>Build a gallery from saved rollouts</summary>

For example, for full Habitat-GS-LN:

```bash
python -m awm.scripts.visualize_evaluation \
  --summary outputs/evaluation/habitatgs_ln/summary.json \
  --episodes data/evaluation/habitatgs_ln/episodes.json \
  --output outputs/evaluation/habitatgs_ln/gallery
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

## Test subsets

Use these selected successful cases to preview navigation and visualizations.
For benchmark scores, use the full sets above. The [data catalog](DATA.md#choose-a-download)
lists both subset sizes and download requirements.

<a id="test-subset"></a>

### Habitat-GS-LN test subset

The Habitat-GS-LN subset contains **16 tasks in 13 scenes**; its network download
is **991.32 MB**. The same steps are included in the [README Quick test](../README.md#quick-test).

Install the [scene converter](INSTALL.md#scene-converter), request
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
Run the prepared subset with the [Quick test command](../README.md#run-navigation).
Open `outputs/evaluation/test_subset/gallery/index.html` to view the results.

### OVON test subset

This subset is listed on [ScaleNav Data](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data#ovon-test-subset).
Accept the Matterport academic-use agreement once on the
[OVON subset access page](https://huggingface.co/datasets/Yangyihui/ScaleNav-OVON-Subset).
Access is granted automatically after submitting the form. Then run:

```bash
hf auth login
python scripts/prepare_data.py ovon --subset test_subset \
  --source data/upstream/evaluation/ovon \
  --scenes data/upstream/evaluation/hm3d \
  --output data/evaluation/ovon_test_subset --download
```

This downloads the 7 scenes, original navmeshes, semantic labels and matching task
metadata, plus the original ABotN starting poses: **642.85 MB** downloaded and
**642.90 MB** after preparation. Files are verified against pinned SHA-256 hashes.
No HM3D TAR archives are needed. Existing matching files are reused.
Run the prepared subset:

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

Open `outputs/evaluation/ovon_test_subset/gallery/index.html` to view the results.

<details>
<summary>OVON sources and download sizes</summary>

The test subset mirrors only the required files from the sources below in
[ScaleNav OVON Subset](https://huggingface.co/datasets/Yangyihui/ScaleNav-OVON-Subset).
HM3D assets retain the Matterport agreement; access requires recorded acceptance.
The HM3D-OVON task files retain their upstream MIT license and underlying
Matterport terms. ABotN starting poses are fetched from their original repository.
Source and license notices are saved alongside the scene assets.

| Source | Test-subset download | Full-OVON download |
| --- | ---: | ---: |
| HM3D v0.2 scene assets | 641,719,313 bytes | 5,676,554,240 bytes (two archives) |
| [HM3D-OVON tasks](https://huggingface.co/datasets/nyokoyama/hm3d_ovon) | 108,890 bytes (7 files) | 165,665,450 bytes (archive) |
| [ABotN starting poses](https://huggingface.co/datasets/acvlab/ABotN-Short-Horizon-OVON) | 1,016,534 bytes | 1,016,534 bytes |
| Source and license notices | 2,530 bytes | Included with the upstream archives |
| **Total** | **642,847,267 bytes (642.85 MB)** | **5,843,236,224 bytes (5.84 GB)** |

Totals count file payloads from an empty cache. Hugging Face cache bookkeeping,
model weights and run outputs are separate. The selected tasks and checksums are
in [the OVON annotations](../annotations/evaluation/short_horizon_ovon.json).

To prepare the test subset from existing official TAR archives instead of the
mirror, add `--hm3d-archives "$HM3D_ARCHIVES"` to its preparation command.

</details>
