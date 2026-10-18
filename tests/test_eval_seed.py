from awm.habitat_extensions.eval_seed import resolve_eval_seed


def test_episode_seed_is_invariant_to_local_index():
    first = resolve_eval_seed(
        42,
        "episode",
        scene_id="0019_840447",
        episode_id="seg_00002",
        local_index=0,
    )
    reshuffled = resolve_eval_seed(
        42,
        "episode",
        scene_id="0019_840447",
        episode_id="seg_00002",
        local_index=999,
    )
    assert first == reshuffled
    assert first == 3330518506


def test_episode_seed_changes_with_episode_and_base_seed():
    first = resolve_eval_seed(
        42,
        "episode",
        scene_id="0019_840447",
        episode_id="seg_00002",
        local_index=0,
    )
    other_episode = resolve_eval_seed(
        42,
        "episode",
        scene_id="0019_840447",
        episode_id="seg_10000",
        local_index=0,
    )
    next_base = resolve_eval_seed(
        43,
        "episode",
        scene_id="0019_840447",
        episode_id="seg_00002",
        local_index=0,
    )
    assert first != other_episode
    assert next_base == (first + 1) % (1 << 32)


def test_legacy_seed_modes_remain_available():
    common = {
        "scene_id": "scene",
        "episode_id": "episode",
        "local_index": 5,
    }
    assert resolve_eval_seed(42, "fixed", **common) == 42
    assert resolve_eval_seed(42, "index", **common) == 47
