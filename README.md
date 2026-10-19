<h1 align="center">ScaleNav</h1>

<h3 align="center">Scalable Internet Video Pretraining<br>for Language-Conditioned Local Navigation</h3>

<p align="center">
  <a href="https://zju3dv.github.io/ScaleNav/"><img src="https://img.shields.io/badge/Project-Page-2563eb?logo=googlechrome&amp;logoColor=white" alt="Project page"></a>
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv paper — coming soon" title="Paper link coming soon">
  <a href="https://huggingface.co/Yangyihui/ScaleNav"><img src="https://img.shields.io/badge/Hugging_Face-Model-ffcc4d?logo=huggingface&amp;logoColor=white" alt="Hugging Face model"></a>
  <a href="https://huggingface.co/datasets/Yangyihui/ScaleNav-Data"><img src="https://img.shields.io/badge/Hugging_Face-Data-57b894?logo=huggingface&amp;logoColor=white" alt="Hugging Face data"></a>
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

> **TL;DR:** ScaleNav learns navigation priors from Internet videos through **action-free pretraining**, then transfers them to **language-conditioned local navigation**.

<p align="center">
  <a href="https://zju3dv.github.io/ScaleNav/#demos"><img src="docs/assets/navigation-demo.gif" width="100%" alt="ScaleNav navigation: real robot on the left, simulation on the right"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#checkpoints">Checkpoints</a> ·
  <a href="#data">Data</a> ·
  <a href="#quick-test">Quick test</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#training">Training</a> ·
  <a href="#real-robot">Real robot</a>
</p>

## Installation

<a id="hardware"></a>
<a id="environment"></a>
<a id="simulator-installation"></a>
<a id="build-dependencies"></a>
<a id="build-habitat-gs"></a>
<a id="scene-converter"></a>
<a id="optional-dependencies"></a>
<a id="development"></a>

Use Linux and an NVIDIA GPU with BF16 support. We recommend **24 GB GPU memory**
and **64 GB system RAM** for inference and evaluation.

Follow the [installation guide](docs/INSTALL.md) to set up the `scalenav`
environment, Habitat-GS and the scene converter.

## Checkpoints

| Model | Download | Size |
| --- | --- | ---: |
| ScaleNav | [Hugging Face](https://huggingface.co/Yangyihui/ScaleNav) | 10.02 GB |
| Wan VAE, text encoder and tokenizer | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | 14.20 GB |

Follow the [checkpoint download steps](docs/INSTALL.md#checkpoints) to download
both models and set `CHECKPOINT` and `BASE_MODEL`.

## Data

Choose the data needed for your task. Each preparation guide covers our annotations
and the required upstream assets.

| Data | Test / preview | Full |
| --- | --- | --- |
| Habitat-GS-LN | [16-task test subset](docs/EVALUATION.md#test-subset) | [157-task benchmark](docs/EVALUATION.md#full-habitat-gs-ln) |
| Short-Horizon OVON | [7-task test subset](docs/EVALUATION.md#ovon-test-subset) | [524-task benchmark](docs/EVALUATION.md#full-short-horizon-ovon) |
| Training | [Small subset (for preview)](docs/DATA.md#small-training-subset) | [Full training data](docs/DATA.md#full-training-data) |

See the [data catalog](docs/DATA.md#choose-a-download) for download sizes and disk
requirements. All datasets are also listed on
[ScaleNav Data](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data#choose-a-dataset).

<a id="quick-start"></a>

## Quick test

After [installation](#installation) and [checkpoint downloads](#checkpoints),
run the **Habitat-GS-LN test subset** to view navigation videos and trajectories.
It contains **16 tasks in 13 scenes**, requiring **991.32 MB** of network
downloads and **2.96 GB** on disk after preparation.

### Prepare the test subset

Request [InteriorGS access](https://huggingface.co/datasets/spatialverse/InteriorGS)
and accept its [terms](https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf),
then prepare the data:

```bash
hf auth login
export PATH="$PWD/external/splat-tools/node_modules/.bin:$PATH"
python scripts/prepare_data.py evaluation --subset test_subset \
  --source data/upstream/evaluation --output data/evaluation/test_subset --download
```

### Run navigation

Run the prepared test subset with visualization enabled:

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

Open `outputs/evaluation/test_subset/gallery/index.html` to view the videos and
trajectories. Metrics are saved in `outputs/evaluation/test_subset/summary.json`.
See [evaluation outputs](docs/EVALUATION.md#visualizations) for details.

To try the other benchmark, follow the
[Short-Horizon OVON test subset](docs/EVALUATION.md#ovon-test-subset).
Test subsets contain selected successful cases; use [full evaluation](#evaluation)
for benchmark scores.

<a id="single-image-inference"></a>

For a standalone trajectory prediction from one image, see
[single-image inference](docs/INFERENCE.md#single-image-inference).

## Evaluation

Run the full benchmarks to report success rate (SR) and SPL:

| Benchmark | Tasks | Data preparation | Run evaluation |
| --- | ---: | --- | --- |
| Habitat-GS-LN | 157 | [Prepare](docs/EVALUATION.md#full-habitat-gs-ln) | [Run](docs/EVALUATION.md#run-habitat-gs-ln-evaluation) |
| Short-Horizon OVON | 524 | [Prepare](docs/EVALUATION.md#full-short-horizon-ovon) | [Run](docs/EVALUATION.md#run-short-horizon-ovon-evaluation) |

The [evaluation guide](docs/EVALUATION.md) covers commands, metrics, visualizations
and the evaluation protocol.

## Training

Prepare the [training data](docs/DATA.md#full-training-data) and
[latent caches](docs/DATA.md#build-training-caches), then follow the
[four-phase training recipe](docs/TRAINING.md#run-the-four-phases).
Use the [small training subset](docs/DATA.md#small-training-subset) to preview the pipeline.
The [training guide](docs/TRAINING.md) covers hardware, schedules and checkpoint resume.

## Real robot

Start the policy server on the GPU machine:

```bash
awm serve --checkpoint "$CHECKPOINT" --base-model "$BASE_MODEL" --decode-video
```

Open `http://127.0.0.1:8620/` for live camera, prediction and trajectory views.
Follow the [Go2 setup](docs/INFERENCE.md#robot-adapter) to connect the camera and
controller, or use the [HTTP API](docs/INFERENCE.md#http-api) with another platform.

## License

Code: [Apache-2.0](LICENSE). Model weights:
[CC BY-NC 4.0](https://huggingface.co/Yangyihui/ScaleNav/blob/main/LICENSE).
Annotations and released Internet videos:
[CC BY-NC-SA 4.0](annotations/LICENSE).
See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and license scope.
