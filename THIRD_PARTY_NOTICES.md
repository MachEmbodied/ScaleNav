# Third-party components

ScaleNav code is licensed under [Apache-2.0](LICENSE). Copyright 2026 ScaleNav
Authors. Upstream copyright and license notices are retained in the model code.

- Wan2.2: backbone architecture and separately supplied VAE/text/tokenizer weights.
  Upstream: https://github.com/Wan-Video/Wan2.2
- Hugging Face Diffusers: model/configuration interfaces, schedulers, and adapted
  pipeline/model utilities. Upstream: https://github.com/huggingface/diffusers
- Hugging Face Transformers: tokenizer and supporting text/model utilities.
  Upstream: https://github.com/huggingface/transformers
- PyTorch / TorchVision: tensor operations, neural network runtime, image utilities.
  Upstream: https://github.com/pytorch/pytorch
- Habitat-Sim / Habitat-GS: optional external simulation runtime. Simulator sources,
  scene data, and compiled binaries are not distributed by this package.

## Data annotations

Copyright 2026 ScaleNav Authors. Files under `annotations/` use
[CC BY-NC-SA 4.0](annotations/LICENSE). Attribute the
[ScaleNav authors](https://github.com/MachEmbodied/ScaleNav), indicate changes,
and license adapted annotations under the same terms.

- **SpatialVID**, Jiahao Wang, Yufeng Yuan and collaborators (CVPR 2026):
  [project](https://github.com/NJU-3DV/SpatialVID). Data uses CC BY-NC-SA 4.0.
  ScaleNav adds navigation instructions, temporal selections and fixed splits,
  and separately distributes prepared 512×288 / 4 Hz Internet videos on
  Hugging Face under the same license. Retain the
  [source and modification notice](annotations/SPATIALVID_NOTICE.txt).
- **InteriorGS**, SpatialVerse Research Team, Manycore Tech Inc. (2025):
  [dataset](https://huggingface.co/datasets/spatialverse/InteriorGS).
  ScaleNav adds instructions, sampled trajectories, actions and splits referencing
  scene identifiers. Separately obtained scenes and geometry annotations retain
  the [InteriorGS terms](https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf).
- **HM3D / HM3D-OVON**: separately obtained scene meshes, semantic annotations
  and category goals. A seven-scene subset and matching MIT-licensed HM3D-OVON
  task files are listed as a [ScaleNav Data subset](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data#ovon-test-subset)
  and hosted in its [access-controlled file repository](https://huggingface.co/datasets/Yangyihui/ScaleNav-OVON-Subset),
  under the original Matterport agreement with recorded acceptance before access.
  The mirror retains upstream file hashes, attribution and license notices.
  Follow the [HM3D access instructions](https://github.com/matterport/habitat-matterport-3dresearch)
  and [Matterport academic-use terms](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data).
  Official task annotations: [HM3D-OVON](https://huggingface.co/datasets/nyokoyama/hm3d_ovon).
- **Habitat-GS navigation meshes**: 45 original evaluation navmeshes are hosted
  in [ScaleNav Data](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data/tree/main/evaluation/habitatgs_ln)
  with the Habitat-GS authors' permission. They retain their upstream terms and
  attribution; the ScaleNav annotation license does not relicense these files.
  See the accompanying [source notice](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data/blob/main/evaluation/habitatgs_ln/SOURCE.txt).
- **ABotN Short-Horizon OVON**: separately downloaded
  [first-visible starting poses](https://huggingface.co/datasets/acvlab/ABotN-Short-Horizon-OVON).
  ScaleNav supplies its 524-task selection, rewritten instructions and evaluation
  settings. Upstream poses, scene geometry and generated target footprints are
  not bundled with these labels.
- **PlayCanvas SplatTransform** (MIT): optional external scene converter.

[Data access instructions](docs/DATA.md#upstream-access-and-terms) cover the
compact Internet video release and separately obtained scenes and Habitat-GS
assets. Gaussian scenes, InteriorGS bounding boxes, scene-rendered RGB and latent
caches are outside this release. HM3D assets in the dedicated OVON subset retain
the Matterport terms; the ScaleNav annotation license does not relicense them.

The code and annotation licenses do not cover model weights or upstream assets.
Third-party dependencies retain their original licenses.
