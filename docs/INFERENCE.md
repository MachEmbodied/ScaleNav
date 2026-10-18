# Inference

Use the model paths from the [README](../README.md#checkpoints).
Model geometry, history and normalization are recovered
from the checkpoint's configuration and deployment metadata.

## Single-image inference

Predict nine navigation actions from an image and instruction and visualize
their trajectory. For continuous navigation with observations after each action, use
[simulation evaluation](EVALUATION.md) or the [robot adapter](#robot-adapter).

After running the [Short-Horizon OVON test subset](EVALUATION.md#run-short-horizon-ovon-evaluation),
extract the first frame and its matching instruction from the coffee-machine case:

```bash
python scripts/prepare_data.py example --source outputs/evaluation/ovon_test_subset \
  --episode 00800-TEEsavR23oF/traj_2411 --output data/example
```

This writes `observation.png`, `instruction.txt` and `source.json`, which records
the episode, instruction and source checksums. The instruction is
**“Walk to the coffee machine visible in your initial view.”** For your own image,
place it at `data/example/observation.png` and save its matching instruction in
`data/example/instruction.txt`.

```bash
awm infer --checkpoint "$CHECKPOINT" --base-model "$BASE_MODEL" \
  --image data/example/observation.png \
  --prompt "$(cat data/example/instruction.txt)" --seed 1065612570 \
  --output outputs/example
```

Open `outputs/example/observation_trajectory.png` to view the input observation
alongside a compact trajectory plot with the predicted path length in metres,
start, final position and heading. The raw action deltas are saved as
`observation_trajectory.npy` in the same directory.

The model uses 288×512 observations at 4 Hz, nine-frame chunks, four sampling
steps, video/action guidance 5/1 and scheduler shifts 5/1.

## History and memory

Single-image inference starts without history. Policy serving retains the initial
RGB anchor and two recent chunks of observed images, together with predicted
actions. The last action of a chunk advances to the first image of the next.
Normalization is symlog with c=0.01. The current observation remains clean during
denoising; video and action use separate schedules.

The default `model_cpu_offload` keeps the text encoder on CPU and the transformer
and VAE on GPU. `model_full_load` also puts the text encoder on GPU and needs more
GPU memory. Check available GPU memory and your process/container RAM limit;
network filesystem reads can make startup slow. Sequential layer offload is not supported.

Policy serving enables history KV caching, batched CFG and CUDA graphs, with video
decoding disabled. Use `--no-cuda-graph` for eager execution or `--decode-video`
for optional video previews. Single-image inference also leaves video decoding
disabled unless `--decode-video` is supplied.

## Outputs

For an input image whose stem is `frame`, a normal single-image call writes:

- `frame_trajectory.npy`: denormalized per-frame deltas, shape `(9, 3)`.
  Columns are raw training-convention `(dx, dy, dyaw)`, in metres/metres/radians.
- `frame_trajectory.png`: input observation and accumulated predicted path.

To plot an existing action file separately:

```bash
python -m awm.scripts.visualize_trajectory \
  --image data/example/observation.png \
  --trajectory outputs/example/observation_trajectory.npy \
  --output outputs/example/observation_trajectory.png
```

The plot uses equal-scale right/forward axes in metres, a minimum 3 m × 3 m view,
and marks the predicted final position and heading. It shows the predicted
actions for this chunk; navigation rollouts are visualized through
[evaluation](EVALUATION.md#visualizations).

Add `--decode-video` to `awm infer` to also save `frame_generated_raw.mp4`
(9 predicted RGB frames at 288×512 / 4 fps).

Do not feed visualization coordinates or cumulative trajectories directly to a
controller expecting per-step deltas or velocities.

## HTTP API

Start the policy server with the model paths from the README:

```bash
awm serve --checkpoint "$CHECKPOINT" --base-model "$BASE_MODEL" --decode-video
```

Open `http://127.0.0.1:8620/` for the dashboard: robot camera,
predicted video, current predicted trajectory, measured trajectory and basic status.
Omit `--decode-video` to skip generated video previews.
One process holds one active instruction/episode; callers must serialize requests.

| Endpoint | Request | Purpose |
| --- | --- | --- |
| `GET /health` | none | Process/policy health |
| `POST /reset` | `{"instruction": "<instruction matching the initial observation>"}` | Start/reset an episode |
| `POST /act` | `{"rgb_b64": "<base64 JPEG or PNG>"}` | One observed frame, next displacement/stop state |
| `POST /act_chunk` | `rgb_b64`, `observations_b64` | Executable chunk; supply intervening observed frames |
| `GET /` | none | Built-in observation/trajectory visualization |
| `POST /telemetry` | `run_id`, `pose`, `state`, `steps`, optional `rgb_b64` | Measured robot pose, execution status and live camera |

`/reset` returns a `run_id`; include it in telemetry for that episode.
`pose` is `[x_m, y_m, yaw_rad]` in a fixed odometry frame, with counterclockwise
yaw. Send `null` when no fresh pose is available. `state` is `waiting`,
`executing`, `model_stop`, `budget`, `error` or `idle`; `steps` counts completed
actions. The actual trajectory starts at the first received pose. The predicted
trajectory starts at the current inference input; the two panels have separate origins.

`/act` can return `delta: [dx_m, dy_m, dyaw_rad]`, `stop`, `stop_after_action`, and `confirm_stop`.
`stop_after_action=true` means execute the accompanying final delta and then halt.
The robot-serving stop rule uses nine consecutive issued actions with planar
magnitude below 0.0325 m and absolute yaw below 0.0325 rad, no earlier than
10 executed actions. Confirmation is disabled by default.
Respect the stop/confirmation fields instead of treating every response as a
motion command. `dt` is the configured control interval; displacements are not
velocities. Responses retain the raw training action convention. The `scan2d_body`
visualization maps raw channels to forward/left/counterclockwise using signs
(+1, -1, -1); this display mapping does not transform the API response. Verify
lateral/yaw signs and timing against the controller before consuming deltas.

The default observed-history mode expects a frame at each `/act` step and reuses
model-generated action history. Chunk mode accepts `observations_b64`: the RGB frames acquired before actions
2 through N of the previous chunk, in execution order. The first request sends
an empty list. The current `rgb_b64` starts the next chunk. This preserves observed
visual history and predicted action history at the same cadence as `/act`.
`stop_after_chunk=true` means execute the returned chunk and halt.
The optional `--history predicted` mode omits intervening observations.
Do not switch modes midway through an episode.

```bash
python examples/policy_client.py --image data/example/observation.png \
  --prompt "$(cat data/example/instruction.txt)"
```

The example resets the episode, submits one image, and prints JSON. It does not
send any robot/controller commands. Images remain in the server's per-process
observation/visualization state. The server has no authentication; use a trusted local connection, SSH tunnel
or authenticated proxy. Keep the raw service off the public internet.

## Robot adapter

Start the [policy server](#http-api) in its default observed-history mode. The client requests a
chunk, executes each displacement over 0.25 seconds, and holds while waiting
for the next prediction. It returns intervening camera frames with the next
request. It stops after at most 200 actions or the model's stop signal.

In another terminal with `scalenav` active, run a hardware-free smoke test with
the paired image and instruction from [single-image inference](#single-image-inference):

```bash
python -m awm.robot --backend dry-run --image data/example/observation.png \
  --instruction "$(cat data/example/instruction.txt)" --max-steps 18
```

For a Go2, complete the README's [Unitree SDK2 setup](../README.md#optional-dependencies).
The adapter uses its `SportClient.Move` and `StopMove` interfaces. Supply the
robot's network interface and a rectified RGB camera stream matching the
approximately 70° horizontal field of view. Set `ROBOT_INSTRUCTION` to a target
description matching the robot's own initial camera view:

```bash
python -m awm.robot --backend go2 --interface eth0 --camera 0 \
  --instruction "${ROBOT_INSTRUCTION:?Set an instruction matching the robot initial view}"
```

Replace `eth0` with the network interface connected to the robot, and `0` with
the camera index or stream path. The client defaults to `http://127.0.0.1:8620`;
when running on a separate machine, use an SSH tunnel and pass its local endpoint
with `--server`.

The adapter reads `position` and `imu_state.rpy` from the official
[`rt/sportmodestate` stream](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_sport_client.cpp)
and sends camera, pose and status updates on a separate thread. Missing or stale
odometry appears as “Waiting for odometry” or “No recent odometry”; the dry-run backend supplies no odometry.

The released checkpoint's raw channels are forward/right/clockwise. Conversion
to Go2 forward/left/counterclockwise velocity is
`(dx, -dy, -dyaw) / 0.25`. The client validates the full returned chunk before
executing it and sends zero velocity on completion or an exception.
Camera calibration and the physical robot are supplied by the deployment environment.

### Offline recording

Add `--record-dir outputs/robot-recordings` to `awm serve` to save each episode
locally. Each run contains `run.json`, timestamped `events.jsonl`, camera/input
images and lossless predicted video arrays (`.npy`). The log associates images,
predictions, actions and measured poses. Recording is disabled by default.

## Checkpoint export

The ScaleNav release contains `ar_model.pt`, `train_meta.pt`, `SHA256SUMS.json`,
`transformer/config.json` and `transformer/diffusion_pytorch_model.safetensors`.
Its [download manifest](checkpoint-download-manifest.json) pins file sizes and hashes.
To export a portable copy from the original approved checkpoint:

```bash
awm export-checkpoint --checkpoint /path/to/original --output /path/to/export
awm check --checkpoint /path/to/export --sha256
```

The exporter verifies source hashes, preserves tensor bytes and removes internal
training paths. It requires about 10.02 GB and refuses to overwrite an existing
directory. Exported metadata is for deployment, not optimizer resume; see
[training](TRAINING.md) for complete-state checkpoints.
