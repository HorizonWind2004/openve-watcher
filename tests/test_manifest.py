from openve_watcher import manifest


def test_sample_meta_round_trip():
    meta = manifest.SampleMeta(base="0000_x", edited_type="global_style", prompt="把天空变紫 {}")
    assert manifest.SampleMeta.from_json(meta.to_json()) == meta


def test_sample_meta_rejects_missing_field():
    try:
        manifest.SampleMeta.from_json('{"base": "a", "edited_type": "b"}')
    except ValueError as exc:
        assert "prompt" in str(exc)
    else:
        raise AssertionError("缺字段时应当报错")


def test_complete_samples_requires_all_three():
    files = [
        "run.json",
        "samples/global_style/a/meta.json",
        "samples/global_style/a/original.mp4",
        "samples/global_style/a/edited.mp4",
        # b 少了 original，是上传中断留下的半个样本
        "samples/local_add/b/meta.json",
        "samples/local_add/b/edited.mp4",
        # 派生文件不参与判定
        "samples/global_style/a/edited_compare.mp4",
    ]
    assert manifest.complete_samples(files) == {("global_style", "a"): "samples/global_style/a"}


def test_parse_sample_dir_ignores_non_samples():
    assert manifest.parse_sample_dir("scores/x.jsonl") is None
    assert manifest.parse_sample_dir("run.json") is None
    assert manifest.parse_sample_dir("samples/t/b/meta.json") == ("t", "b")


def test_score_path_encodes_protocol():
    path = manifest.score_path("gemini-2.5-pro")
    assert path == "scores/gemini-2.5-pro_gemini_video_score.jsonl"
    assert "gemini_video" in path, "协议要写进文件名，避免和抽帧代理旧分数混用"


def test_score_path_sanitises_slash():
    assert "/" not in manifest.score_path("models/gemini").split("/")[-1]
