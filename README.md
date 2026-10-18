<h1 align="center">ScaleNav</h1>

<h3 align="center">Scalable Internet Video Pretraining<br>for Language-Conditioned Local Navigation</h3>

<p align="center">
  <a href="https://yangyihui0502.github.io/scalenav-review/"><img src="https://img.shields.io/badge/Project-Page-2563eb?logo=googlechrome&amp;logoColor=white" alt="Project page"></a>
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv paper — coming soon" title="Paper link coming soon">
  <a href="https://huggingface.co/Yangyihui/ScaleNav"><img src="https://img.shields.io/badge/Hugging_Face-Model-ffcc4d?logo=huggingface&amp;logoColor=white" alt="Hugging Face model"></a>
</p>

<p align="center">
  <a href="https://yangyihui0502.github.io/">Yihui Yang</a><sup>1</sup> ·
  <a href="https://yuanhongyu.xyz/">Yuanhong Yu</a><sup>1</sup> ·
  <a href="https://pluswave.top/">Jiawei Wang</a><sup>1</sup> ·
  Zhongrui Yu<sup>2</sup> ·
  Jia Shi<sup>2</sup><br>
  <a href="https://zhankunliauto.github.io/">Kun Zhan</a><sup>2</sup> ·
  <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a><sup>3</sup> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a><sup>1</sup> ·
  Yu&nbsp;Liu<sup>2</sup> ·
  <a href="https://pengsida.net/">Sida Peng</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>&nbsp;Zhejiang University &nbsp; <sup>2</sup>&nbsp;Li Auto Inc. &nbsp; <sup>3</sup>&nbsp;Shenzhen University
</p>

<p align="center">
  <a href="https://yangyihui0502.github.io/scalenav-review/#demos"><img src="docs/assets/navigation-demo.gif" width="100%" alt="ScaleNav navigation: real robot on the left, simulation on the right"></a>
</p>

> **TL;DR:** ScaleNav learns navigation priors from Internet videos through **action-free pretraining**, then transfers them to **language-conditioned local navigation**.

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#checkpoints">Checkpoints</a> ·
  <a href="#data">Data</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#training">Training</a> ·
  <a href="#real-robot">Real robot</a>
</p>

## Installation

### Hardware

Use Linux and an NVIDIA GPU with BF16 support.

| Workload | Recommended GPU memory | Configuration |
| --- | --- | --- |
| Navigation evaluation and policy inference | One 24 GB GPU | BF16, 512×288 observations, default `model_cpu_offload` |
| Action-alignment training | 48 GB per GPU | `--micro-batch 1 --deepspeed`; optimizer states are offloaded to CPU |

The 16-task Habitat-GS-LN test subset used **18.4 GiB peak GPU memory** in a
48 GB reference run, sampled with `nvidia-smi` and visualization enabled.

Allow **64 GB of system RAM** for evaluation. The default inference mode keeps
the text encoder on CPU and the transformer and VAE on GPU. GPU memory also
includes simulator scenes, history caches and CUDA graphs.

For full-parameter training without sharding or offload, FP32 parameters,
gradients and Adam states alone occupy about **75 GiB**, before activations
and the text encoder. Use
[`--deepspeed`](#optional-dependencies) for CPU optimizer offload and
`--text-device cpu` to move the frozen text encoder off the GPU. CPU offload
requires additional system RAM. For single-process action alignment, **128 GB
of system RAM is recommended**, including checkpoint saves. Budget additional
RAM for multiple training processes using the [DeepSpeed memory estimator](https://deepspeed.readthedocs.io/en/latest/memory.html#discussion).

### Environment

Create the Python 3.12 environment:

```bash
conda create -n scalenav --override-channels -c conda-forge python=3.12 pip ffmpeg git
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scalenav

git -c http.sslBackend=openssl clone https://github.com/MachEmbodied/ScaleNav.git
cd ScaleNav
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[inference,eval,training]'
awm doctor --require-cuda --data-tools
```

Install [Miniforge](https://github.com/conda-forge/miniforge#install) if Conda is
unavailable.

### Build dependencies

Install the compiler and CUDA toolkit for Habitat-GS and DeepSpeed in the same
`scalenav` environment:

```bash
conda activate scalenav
conda install --override-channels -c conda-forge \
  gcc_linux-64=12 gxx_linux-64=12 libegl-devel libgl-devel libjpeg-turbo zlib glm=0.9.9.8
conda install --override-channels -c nvidia/label/cuda-12.4.1 -c conda-forge cuda-toolkit=12.4.1
conda activate scalenav

python -m pip install cmake==3.31.10 ninja scikit-build-core pybind11 setuptools wheel
```

### Simulator installation

Both evaluation benchmarks and simulation training-data preparation require
[Habitat-GS](https://github.com/zju3dv/habitat-gs), which provides the `habitat_sim`
Python module. After setting up the [build dependencies](#build-dependencies),
run from the ScaleNav repository root:

```bash
(
  set -e
  conda activate scalenav
  export PYTHONNOUSERSITE=1
  export CUDA_HOME="${CONDA_PREFIX:?Activate the scalenav Conda environment}"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export CUDAHOSTCXX="${CXX:?Install the build dependencies and reactivate scalenav}"
  export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export CMAKE_BUILD_PARALLEL_LEVEL=2
  python -c 'import os, sys; assert sys.prefix == os.environ["CONDA_PREFIX"], "Python must come from the active Conda environment"; print(sys.executable)'
  python -m awm.cli doctor --require-cuda --data-tools
  "$CUDACXX" --version
  "$CUDAHOSTCXX" --version

  scalenav_git_slot=${GIT_CONFIG_COUNT:-0}
  export "GIT_CONFIG_KEY_${scalenav_git_slot}=http.sslBackend"
  export "GIT_CONFIG_VALUE_${scalenav_git_slot}=openssl"
  export GIT_CONFIG_COUNT=$((scalenav_git_slot + 1))

  if [ ! -d external/habitat-gs ]; then
    git clone https://github.com/zju3dv/habitat-gs.git external/habitat-gs
  fi
  git -C external/habitat-gs checkout cb01b2fdf850ce3cab28ad00461a68ac7719f23f
  git -C external/habitat-gs submodule update --init --recursive
  HABITAT_BUILD_GUI_VIEWERS=OFF HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF \
    python -m pip install ./external/habitat-gs --no-build-isolation \
    --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native
  python -m awm.cli doctor --require-cuda --data-tools --simulator
)
```

### Scene converter

Habitat-GS-LN evaluation and simulation training-data preparation also need the
InteriorGS scene converter. From the repository root:

```bash
conda install --override-channels -c conda-forge nodejs=22
npm install --prefix external/splat-tools @playcanvas/splat-transform@3.10.1
export PATH="$PWD/external/splat-tools/node_modules/.bin:$PATH"
```

### Optional dependencies

**DeepSpeed:** for training with `--deepspeed`, complete the
[build dependencies](#build-dependencies), then install:

```bash
python -m pip install -e '.[distributed]'
```

DeepSpeed builds optimizer extensions on first use. In a new terminal, activate
`scalenav` and set `CUDA_HOME="$CONDA_PREFIX"` before training.

**Go2 robot:** install the official
[Unitree SDK2 Python package](https://github.com/unitreerobotics/unitree_sdk2_python#installation)
in the environment running the robot adapter. Policy serving and the dry-run
adapter use the base environment above.

**Development:** install the test and package-building tools:

```bash
python -m pip install -e '.[dev]'
```

## Checkpoints

| Model | Download | Size |
| --- | --- | --- |
| ScaleNav | [Hugging Face](https://huggingface.co/Yangyihui/ScaleNav) | 10.02 GB |
| Wan VAE, text encoder and tokenizer | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | 14.20 GB |

Download ScaleNav and its Wan components. The private model repository requires
an authorized Hugging Face account:

```bash
hf auth login
python scripts/download_checkpoint.py --output checkpoints/scalenav
awm check --checkpoint checkpoints/scalenav --sha256

python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
    local_dir="checkpoints/Wan2.2-TI2V-5B",
    allow_patterns=["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/*"],
)
PYTHON

export CHECKPOINT=checkpoints/scalenav
export BASE_MODEL=checkpoints/Wan2.2-TI2V-5B
awm doctor --checkpoint "$CHECKPOINT" --base-model "$BASE_MODEL" --require-cuda
```

The downloader uses a pinned Hugging Face revision, resumes interrupted downloads
and verifies SHA-256. Existing verified files are reused. For training from P0, omit
`allow_patterns` to download the full Wan model (34.20 GB).

## Data

Access to [ScaleNav Data](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data)
is currently required for the Internet video packs and Habitat-GS navmeshes.

Each **Download** link leads to the complete preparation steps for that dataset:
ScaleNav annotations, required media or official scenes, and the resulting data
directory. Total data storage counts the prepared data directories and retained
source downloads from all sources. Model weights, environments,
training latent caches and run outputs are separate. Sizes sum file bytes and use
decimal MB/GB.

| Dataset | Contents | Total data storage | Download |
| --- | --- | ---: | --- |
| Habitat-GS-LN test subset | 16 tasks across 13 scenes | 2.96 GB | [Download](docs/EVALUATION.md#test-subset) |
| Full Habitat-GS-LN evaluation | 157 tasks across 45 scenes | 10.36 GB | [Download](docs/EVALUATION.md#full-habitat-gs-ln) |
| Short-Horizon OVON test subset | 7 tasks across 7 scenes | 11.52 GB | [Download](docs/EVALUATION.md#short-horizon-ovon) |
| Full Short-Horizon OVON evaluation | 524 tasks across 36 scenes | 11.54 GB | [Download](docs/EVALUATION.md#short-horizon-ovon) |
| Small training subset (for preview) | 17 Internet segments and 4 simulated episodes | 239.54 MB | [Download](docs/DATA.md#small-training-subset) |
| Full training data | 18,249 Internet segments and 2,600 simulated episodes | 153.04 GB | [Download](docs/DATA.md#full-training-data) |

## Evaluation

Complete [simulator installation](#simulator-installation) and prepare the
corresponding evaluation data from the table above.

**Habitat-GS-LN** — use `EVAL_SET=test_subset` for the
[16-task test subset](docs/EVALUATION.md#test-subset), or `EVAL_SET=habitatgs_ln`
for the [full 157-task benchmark](docs/EVALUATION.md#full-habitat-gs-ln):

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

**Short-Horizon OVON** — use `EVAL_SET=ovon_test_subset` for the
[7-task test subset](docs/EVALUATION.md#short-horizon-ovon), or
`EVAL_SET=short_horizon_ovon` for all 524 tasks:

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

Each output directory contains `summary.json`: `aggregate.success_rate` is SR,
`aggregate.spl` is SPL, and `episodes` contains per-task results.
`--visualize` saves every case's video and a `gallery/index.html` page with
synchronized trajectories, targets and outcomes. Omit it for metrics only. See the
[evaluation protocol](docs/EVALUATION.md#evaluation-protocol) for metric definitions.

To visualize a predicted trajectory from one image and instruction, see
[single-image inference](docs/INFERENCE.md#single-image-inference).

## Training

The [environment setup](#environment) includes training dependencies.
Prepare the data using the [data guide](docs/DATA.md), then set its location:

```bash
export TRAIN_DATA="${TRAIN_DATA:-data/training/goto}"
```

Follow [training](docs/TRAINING.md#run-the-four-phases) for
P0 → P1 → action alignment → joint fine-tuning. Use
`TRAIN_DATA=data/smoke/goto` with the small subset to test the pipeline.
The recipe and model settings are in [src/awm/configs](src/awm/configs).

## Real robot

Start the policy server on the GPU machine:

```bash
awm serve --checkpoint "$CHECKPOINT" --base-model "$BASE_MODEL" --decode-video
```

Open `http://127.0.0.1:8620/` for the camera, prediction and measured-trajectory
viewer. The Go2 adapter sends camera observations to the policy, executes action
chunks and stops on the model's stop signal or after 200 actions.

Install [Unitree SDK2](#optional-dependencies) and follow the
[robot setup](docs/INFERENCE.md#robot-adapter) to connect a rectified RGB camera.
In a second terminal with the `scalenav`
environment active, set `ROBOT_INSTRUCTION` to a description matching the robot's
initial view, replace `eth0` and `0` with the robot interface and camera, then run:

```bash
python -m awm.robot --backend go2 --interface eth0 --camera 0 \
  --instruction "${ROBOT_INSTRUCTION:?Set an instruction matching the robot initial view}"
```

This uses the local policy server. For a separate client machine, connect through
an SSH tunnel and set `--server` to the forwarded local address. See the
[robot adapter](docs/INFERENCE.md#robot-adapter) for the hardware-free check and
[HTTP API](docs/INFERENCE.md#http-api) for integrating another platform.

## Development

After installing the [development dependencies](#optional-dependencies):

```bash
python -m pytest
python -m build
python scripts/check_release.py
```

## License

Code: [Apache-2.0](LICENSE). Annotations and released Internet videos:
[CC BY-NC-SA 4.0](annotations/LICENSE).
See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and license scope.
