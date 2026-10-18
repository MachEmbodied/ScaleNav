import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from awm.data.datasets.nav_t2v import NavT2VDataset
from awm.scripts import precompute_nav_t2v_latents as video_cache
from awm.scripts import precompute_nav_wam_latents as action_cache


class TinyVAE:
    def encode(self, video):
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: video[:, :1, ::4, :2, :2]))


def write_video(path, frames=11):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 4, (16, 16))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((16, 16, 3), i * 8, dtype=np.uint8))
    writer.release()


def test_action_chunks_keep_ninth_transition_and_pad_physical_zeros(tmp_path):
    write_video(tmp_path / 'video.mp4')
    actions = np.arange(30, dtype=np.float32).reshape(10, 3) / 100
    np.save(tmp_path / 'actions.npy', actions)
    args = SimpleNamespace(data_root=str(tmp_path), video_sample_n_frames=9,
                           video_sample_stride=1, chunk_overlap_frames=0, chunk_mode='ceil',
                           video_sample_size=[16, 16], cam_pitch_default=0., encode_batch_size=1)
    row = dict(key='scene/episode', scene='scene', episode='episode', video='video.mp4',
               actions='actions.npy', canonical='Go to the chair.')
    output = tmp_path / 'cache'
    result = action_cache.encode_episode(args, row, 0, TinyVAE(), torch.float32,
                                         torch.device('cpu'), output)
    assert result['n_chunks'] == 2
    first = torch.load(output / 'scene/episode/chunk_0000.pt', weights_only=True)
    last = torch.load(output / 'scene/episode/chunk_0001.pt', weights_only=True)
    torch.testing.assert_close(first['trajectory'], torch.from_numpy(actions[:9]))
    torch.testing.assert_close(last['trajectory'][0], torch.from_numpy(actions[9]))
    assert last['trajectory'][1:].count_nonzero() == 0
    assert last['real_len_in_output'] == 2
    assert last['inpaint_latent'].shape[1] == 1


@pytest.mark.parametrize('real_frames,is_last', [(0, True), (2, True), (9, True), (9, False)])
def test_cached_action_padding_keeps_stop_supervision(real_frames, is_last):
    from awm.data.datasets.nav_wam_cached import NavWamCachedDataset

    dataset = NavWamCachedDataset.__new__(NavWamCachedDataset)
    trajectory = torch.zeros(9, 3)
    trajectory[:real_frames] = 0.1
    original = trajectory.clone()
    result, action_mask, video_mask = dataset._build_stop_and_mask(
        trajectory, is_last, real_frames,
    )
    torch.testing.assert_close(result, original)
    torch.testing.assert_close(action_mask, torch.ones(9))
    expected_video = (torch.arange(9) < real_frames).float() if is_last else torch.ones(9)
    torch.testing.assert_close(video_mask, expected_video)


def test_invalid_video_window_does_not_fall_back_to_unrelated_frames(tmp_path):
    write_video(tmp_path / 'video.mp4', frames=20)
    dataset = NavT2VDataset.__new__(NavT2VDataset)
    dataset.min_required_frames = 9
    with pytest.raises(ValueError):
        dataset._load_window(str(tmp_path / 'video.mp4'), 18, 30)


def test_video_preparation_failure_is_not_marked_complete(tmp_path, monkeypatch):
    config = tmp_path / 'data.yaml'
    config.write_text('datasets: []\n')
    (tmp_path / 'Wan2.2_VAE.pth').write_bytes(b'test encoder')
    args = SimpleNamespace(config=str(config), output_dir=str(tmp_path / 'cache'),
                           device='cpu', video_sample_n_frames=9, video_sample_stride=1,
                           video_sample_size=[16, 16], chunk_overlap_frames=0, encode_batch_size=1,
                           pretrained_model_path=str(tmp_path), dtype='float32')
    dataset = SimpleNamespace(required_raw_frames=9, chunk_step_raw_frames=9)
    class Source:
        samples = [{'video': 'first.mp4'}, {'video': 'corrupt.mp4'}]
        required_raw_frames = 9
        chunk_step_raw_frames = 9
        def __len__(self):
            return 2
    monkeypatch.setattr(video_cache, 'parse_args', lambda: args)
    monkeypatch.setattr(video_cache, 'build_source_dataset', lambda *values: Source())
    monkeypatch.setattr(video_cache, 'load_vae', lambda *values: (TinyVAE(), torch.float32))
    def encode(*values):
        if values[2] == 1:
            raise ValueError('corrupt input video')
        return {'source_index': 0, 'n_chunks': 1}
    monkeypatch.setattr(video_cache, 'encode_sample', encode)
    with pytest.raises(RuntimeError, match='incomplete'):
        video_cache.main()
    manifest = json.loads((tmp_path / 'cache/manifest.json').read_text())
    assert manifest['complete'] is False
    assert manifest['cached_samples'] == 1


def test_video_cache_rejects_changed_annotations_or_encoder(tmp_path):
    config = tmp_path / 'data.yaml'
    config.write_text('datasets: []\n')
    weights = tmp_path / 'Wan2.2_VAE.pth'
    weights.write_bytes(b'original encoder')
    args = SimpleNamespace(config=str(config), pretrained_model_path=str(tmp_path),
                           dtype='bfloat16', video_sample_n_frames=9, video_sample_stride=1,
                           video_sample_size=[16, 16], chunk_overlap_frames=0)
    source = SimpleNamespace(required_raw_frames=9, chunk_step_raw_frames=9,
                             samples=[{'video': 'clip.mp4', 'text': 'Go left.'}])
    output = tmp_path / 'cache'
    output.mkdir()
    video_cache.ensure_cache_spec(args, source, output, 0, 1)
    video_cache.ensure_cache_spec(args, source, output, 0, 1)
    source.samples[0]['text'] = 'Go right.'
    with pytest.raises(RuntimeError, match='mismatch'):
        video_cache.ensure_cache_spec(args, source, output, 0, 1)
    source.samples[0]['text'] = 'Go left.'
    weights.write_bytes(b'replacement encoder')
    with pytest.raises(RuntimeError, match='mismatch'):
        video_cache.ensure_cache_spec(args, source, output, 0, 1)


def test_action_cache_rejects_changed_encoder_or_pitch(tmp_path):
    config = tmp_path / 'model.yaml'
    config.write_text('vae_kwargs: {}\n')
    (tmp_path / 'meta.jsonl').write_text('{}\n')
    weights = tmp_path / 'Wan2.2_VAE.pth'
    weights.write_bytes(b'original encoder')
    args = SimpleNamespace(config=str(config), pretrained_model_path=str(tmp_path),
                           meta_path=str(tmp_path / 'meta.jsonl'), split_file=None,
                           data_root=str(tmp_path), dtype='bfloat16', cam_pitch_default=0.,
                           video_sample_n_frames=9, video_sample_stride=1,
                           video_sample_size=[16, 16], chunk_overlap_frames=0, chunk_mode='ceil')
    output = tmp_path / 'cache'
    output.mkdir()
    action_cache.ensure_spec(action_cache.cache_spec(args, []), output, 0, 1)
    args.cam_pitch_default = .2
    with pytest.raises(RuntimeError, match='mismatch'):
        action_cache.ensure_spec(action_cache.cache_spec(args, []), output, 0, 1)
    args.cam_pitch_default = 0.
    weights.write_bytes(b'replacement encoder')
    with pytest.raises(RuntimeError, match='mismatch'):
        action_cache.ensure_spec(action_cache.cache_spec(args, []), output, 0, 1)


@pytest.mark.parametrize("stage", ["align", "joint"])
def test_action_training_samples_chunks_uniformly_across_unequal_episodes(tmp_path, stage):
    import hashlib
    from collections import Counter
    from awm.training.data import build_sources, UpdateBatchSampler
    from awm.training.recipe import load_recipe

    recipe = load_recipe(stage)
    video_root, action_root = tmp_path / "video", tmp_path / "action"
    (video_root / "sample").mkdir(parents=True)
    torch.save({"latent": torch.zeros(1, 3, 2, 2),
                "inpaint_latent": torch.zeros(1, 1, 2, 2)},
               video_root / "sample/chunk_0000.pt")
    (video_root / "manifest.json").write_text(json.dumps({
        "complete": True, "video_sample_n_frames": 9, "video_sample_stride": 1,
        "video_sample_size": [288, 512], "chunk_overlap_frames": 0, "chunk_step": 9,
        "samples": [{"relative_dir": "sample", "n_chunks": 1, "text": "Approach the chair."}],
    }))
    episodes, labels = [], []
    marker = 0
    for name, chunks in [("short", 1), ("long", 3)]:
        directory = action_root / "scene" / name
        directory.mkdir(parents=True)
        episodes.append({"key": f"scene/{name}", "scene": "scene",
                         "episode": name, "n_chunks": chunks})
        labels.append({"key": f"scene/{name}", "canonical": "Approach the chair."})
        for index in range(chunks):
            torch.save({"latent": torch.full((1, 3, 2, 2), float(marker)),
                        "inpaint_latent": torch.zeros(1, 1, 2, 2),
                        "trajectory": torch.zeros(9, 3), "is_last_chunk": index == chunks - 1,
                        "real_len_in_output": 9}, directory / f"chunk_{index:04d}.pt")
            marker += 1
    meta = tmp_path / "meta.jsonl"
    meta.write_text("\n".join(json.dumps(row) for row in labels) + "\n")
    split = tmp_path / "train.txt"
    split.write_text("scene/short\nscene/long\n")
    (action_root / "manifest.json").write_text(json.dumps({
        "version": 2, "complete": True, "video_sample_size": [288, 512],
        "source_meta_sha256": hashlib.sha256(meta.read_bytes()).hexdigest(), "episodes": episodes,
    }))
    args = SimpleNamespace(video_cache=str(video_root), action_cache=str(action_root),
                           action_meta=str(meta), action_split=str(split))
    sources, _ = build_sources(args, recipe)
    sampler = UpdateBatchSampler([len(source) for source in sources.sources],
                                 replay_probability=0, micro_batch=1,
                                 total_micro_batches=512, seed=42)
    counts = Counter(int(sources[item]["latents"][0, 0, 0, 0])
                     for group in sampler for item in group)
    assert set(counts) == {0, 1, 2, 3}
    assert all(90 <= count <= 166 for count in counts.values()), counts
