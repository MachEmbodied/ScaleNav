import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

from awm.scripts.visualize_trajectory import integrate_actions


def test_actions_use_preceding_body_frame_and_clockwise_yaw():
    path = integrate_actions([[1, 0, np.pi / 2], [1, 0, 0], [0, 1, -np.pi / 2]])
    np.testing.assert_allclose(path[:, :2], [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-15)
    np.testing.assert_allclose(path[:, 2], [0, np.pi / 2, np.pi / 2, 0], atol=1e-15)


@pytest.mark.parametrize('actions', [[], [[1, 2]], [[1, np.nan, 0]], [[0, 0, np.inf]]])
def test_invalid_trajectories_are_rejected(actions):
    with pytest.raises(ValueError):
        integrate_actions(actions)


@pytest.fixture
def rollouts(tmp_path):
    videos = tmp_path / 'videos'
    videos.mkdir()
    video = videos / 'rollout.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 4, (32, 24))
    assert writer.isOpened()
    for color in ([0, 0, 255], [0, 255, 0], [255, 0, 0]):
        writer.write(np.full((24, 32, 3), color, dtype=np.uint8))
    writer.release()
    scene = tmp_path / 'scene.glb'
    scene.touch()
    episodes, rows = [], []
    for index, success in enumerate([True, False]):
        episode = {
            'scene_id': 'scene', 'episode_id': str(index), 'instruction': 'Approach the <desk> & stop.',
            'start_xyz': [0, .35, 0], 'start_quat_wxyz': [1, 0, 0, 0], 'goal_xyz': [0, 0, -2],
            'action_convention': 'scan2d_body_forward_right_clockwise', 'scene_path': str(scene),
            'reference_goal_positions': [[0, 0, -1]],
            'target_contract': {'semantic_target_footprint_world_xz': [[-1, -2], [1, -2], [1, -3], [-1, -3]]},
        }
        episodes.append(episode)
        rows.append({**{k: episode[k] for k in ['scene_id', 'episode_id', 'instruction']},
                     'video_path': str(video), 'executed_positions_world': [[0, .35, 0], [0, .35, -.5], [0, .35, -1]],
                     'success': success, 'semantic_region_success': True, 'stopped_via_model': success,
                     'success_type': 'semantic_contract', 'navigation_error': .5, 'spl': float(success),
                     'path_length': 1., 'steps_taken': 2, 'termination_reason': 'model_stop' if success else 'max_steps'})
    return rows, episodes, videos


@pytest.mark.parametrize('benchmark', ['habitatgs_ln', 'short_horizon_ovon'])
@pytest.mark.parametrize('visualize', [False, True])
def test_evaluation_switch_saves_all_cases_without_changing_metrics(rollouts, tmp_path, monkeypatch, benchmark, visualize):
    from awm.habitat_extensions.awm_habitat_gs_evaluator import AWMHabitatGSEvaluator
    from awm.scripts import eval_habitat_gs
    from awm.scripts.visualize_evaluation import BENCHMARK_TITLES
    rows, episodes, videos = rollouts
    manifest = tmp_path / 'episodes.json'
    manifest.write_text(json.dumps({'benchmark': benchmark, 'episodes': episodes}))
    config_file = Path(__file__).resolve().parents[1] / 'src/awm/configs' / (
        'eval_habitatgs.yaml' if benchmark == 'habitatgs_ln' else 'eval_ovon.yaml')
    monkeypatch.setattr(sys, 'argv', ['eval', '--eval_config', str(config_file),
                                    '--episode_manifest', str(manifest), '--gs_dataset_root', str(tmp_path),
                                    '--model_path', 'model', '--pretrained_model_path', 'base',
                                    '--output_dir', str(tmp_path), '--visualize' if visualize else '--no-visualize'])
    args = eval_habitat_gs.parse_args()
    config, *_ = eval_habitat_gs._build_config(args.eval_config, args)
    evaluator = AWMHabitatGSEvaluator.__new__(AWMHabitatGSEvaluator)
    evaluator.config = config
    evaluator.split_file = manifest
    evaluator._progress_path = tmp_path / 'progress.jsonl'
    evaluator._episode_scene_path = lambda ep: Path(ep['scene_path'])
    evaluator._run_episode = lambda ep: dict(rows[int(ep['episode_id'])])
    aggregate = evaluator.evaluate()
    assert aggregate['n_episodes'] == 2
    assert aggregate['success_rate'] == aggregate['spl'] == .5
    saved = json.loads((tmp_path / 'summary.json').read_text())
    assert saved['benchmark'] == benchmark and len(saved['episodes']) == 2
    assert (tmp_path / 'gallery/index.html').exists() is visualize
    if visualize:
        page = (tmp_path / 'gallery/index.html').read_text()
        assert page.count('<article class="card"') == 2
        assert f'<h1>{BENCHMARK_TITLES[benchmark]}</h1>' in page
        assert '&lt;desk&gt; &amp; stop.' in page
        assert 'Failure · no model STOP' in page
        assert (tmp_path / 'gallery/videos/02.mp4').read_bytes() == (videos / 'rollout.mp4').read_bytes()
        assert (tmp_path / 'gallery/overview.png').is_file()
        with Image.open(tmp_path / 'gallery/frames/01-last.jpg') as image:
            assert image.getpixel((0, 0))[2] > 200


def test_gallery_paginates_overviews_and_can_be_regenerated(rollouts, tmp_path):
    from awm.scripts.visualize_evaluation import build_gallery, read_cases
    rows, episodes, videos = rollouts
    cases = read_cases(rows, episodes, videos)
    output = tmp_path / 'gallery'
    build_gallery(cases, output)
    build_gallery(cases * 10 + cases[:1], output)
    page = (output / 'index.html').read_text()
    assert page.count('<article class="card"') == 21
    for name in ('overview-001.png', 'overview-002.png'):
        assert name in page
        with Image.open(output / name) as image:
            assert image.height < 4000


@pytest.mark.parametrize('ensure_ascii', [True, False])
def test_gallery_cli_supports_utf8_in_an_ascii_locale(rollouts, tmp_path, ensure_ascii):
    rows, episodes, _ = rollouts
    instruction = 'Approach the caf\u00e9 <desk> & stop.'
    rows[0]['instruction'] = episodes[0]['instruction'] = instruction
    summary = tmp_path / 'summary.json'
    manifest = tmp_path / 'episodes.json'
    summary.write_text(json.dumps({'episodes': rows}, ensure_ascii=ensure_ascii), encoding='utf-8')
    manifest.write_text(json.dumps({'episodes': episodes}, ensure_ascii=ensure_ascii), encoding='utf-8')
    output = tmp_path / 'gallery'
    env = dict(os.environ, LC_ALL='C', PYTHONUTF8='0', PYTHONCOERCECLOCALE='0',
               PYTHONIOENCODING='ascii')
    result = subprocess.run(
        [sys.executable, '-m', 'awm.scripts.visualize_evaluation',
         '--summary', str(summary), '--episodes', str(manifest), '--output', str(output)],
        env=env, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')
    page = (output / 'index.html').read_text(encoding='utf-8')
    assert 'caf\u00e9 &lt;desk&gt; &amp; stop.' in page
    assert 'Success \u00b7 model STOP' in page
    assert page.count('<video ') == 2


@pytest.mark.parametrize('invalid', ['instruction', 'frames', 'video'])
def test_gallery_rejects_misaligned_or_missing_rollouts(rollouts, invalid):
    from awm.scripts.visualize_evaluation import read_cases
    rows, episodes, videos = rollouts
    if invalid == 'instruction':
        rows[0]['instruction'] = 'Different task'
    elif invalid == 'frames':
        rows[0]['executed_positions_world'].pop()
    else:
        rows[0]['video_path'] = ''
    with pytest.raises(ValueError):
        read_cases(rows, episodes, videos)
