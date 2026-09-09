import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openve_watcher import manifest, watch


def test_scored_bases_skips_truncated_and_invalid(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"base": "ok", "scores": [3, 4, 5]}) + "\n"
        + '{"base": "truncated", "scores": [3, 4\n'          # 写一半被截断
        + json.dumps({"base": "range", "scores": [0, 4, 5]}) + "\n"   # 越界
        + json.dumps({"base": "short", "scores": [3, 4]}) + "\n"      # 少一项
        + json.dumps({"scores": [3, 4, 5]}) + "\n",                   # 没有 base
        encoding="utf-8",
    )
    assert watch.scored_bases(path) == {"ok"}


def test_scored_bases_missing_file_is_empty(tmp_path):
    assert watch.scored_bases(tmp_path / "nope.jsonl") == set()


def test_load_api_keys_dedupes_preserving_order(tmp_path, monkeypatch):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("# 注释\nk1\nk2\n\nk1\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "k2,k3")
    assert watch.load_api_keys(key_file) == ["k1", "k2", "k3"]


def test_load_api_keys_empty_without_source(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert watch.load_api_keys(None) == []


def test_system_prompt_injects_edit_instruction():
    text = watch.system_prompt("global_style", "把天空变紫", json_mode=False)
    assert "把天空变紫" in text
    assert "{edit_prompt}" not in text


def test_system_prompt_json_mode_appends_instruction():
    text = watch.system_prompt("global_style", "x", json_mode=True)
    assert "instruction_compliance" in text


def test_system_prompt_unknown_type_raises():
    with pytest.raises(KeyError):
        watch.system_prompt("no_such_type", "x", json_mode=False)


class FakeDataset:
    def __init__(self, repo_id):
        self.id = repo_id


class FakeApi:
    def __init__(self, datasets=(), files=()):
        self._datasets = [FakeDataset(d) for d in datasets]
        self._files = list(files)
        self.token = None
        self.uploaded = []

    def list_datasets(self, author, search):
        return list(self._datasets)

    def list_repo_files(self, repo_id, repo_type):
        return list(self._files)

    def upload_file(self, **kwargs):
        self.uploaded.append(kwargs["path_in_repo"])


def test_discover_repos_filters_by_prefix():
    api = FakeApi(["sanaka87/openve_a", "sanaka87/other_b", "sanaka87/openve_c"])
    assert watch.discover_repos(api, "sanaka87", "openve_") == [
        "sanaka87/openve_a",
        "sanaka87/openve_c",
    ]


def _args(tmp_path, **over):
    base = dict(
        out_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        model="gemini-2.5-pro",
        timeout=60.0,
        max_workers=1,
        json_mode=False,
        push_scores=False,
        limit=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_process_repo_scores_only_complete_and_unscored(tmp_path, monkeypatch):
    files = [
        "samples/global_style/a/meta.json",
        "samples/global_style/a/original.mp4",
        "samples/global_style/a/edited.mp4",
        "samples/local_add/b/meta.json",          # 半个样本
        "samples/local_add/b/edited.mp4",
    ]
    api = FakeApi(files=files)
    args = _args(tmp_path)

    def fake_fetch(_api, _repo, edited_type, base, _cache):
        meta = manifest.SampleMeta(base=base, edited_type=edited_type, prompt="p")
        return meta, Path("orig.mp4"), Path("edit.mp4")

    monkeypatch.setattr(watch, "fetch_sample", fake_fetch)
    monkeypatch.setattr(
        "openve_watcher.gemini.evaluate_video_pair",
        lambda *a, **k: ([4, 4, 4], "raw"),
    )

    added, total, done = watch.process_repo(api, "sanaka87/openve_x", args, ["k"])
    assert (added, total, done) == (1, 1, 0), "只有三件套齐全的样本才打分"

    added2, _, done2 = watch.process_repo(api, "sanaka87/openve_x", args, ["k"])
    assert (added2, done2) == (0, 1), "第二轮应当靠 jsonl 断点续跑，不重复打分"


def test_process_repo_survives_single_sample_failure(tmp_path, monkeypatch, capsys):
    files = [
        "samples/t/a/meta.json",
        "samples/t/a/original.mp4",
        "samples/t/a/edited.mp4",
        "samples/t/b/meta.json",
        "samples/t/b/original.mp4",
        "samples/t/b/edited.mp4",
    ]
    api = FakeApi(files=files)
    args = _args(tmp_path)

    def fake_fetch(_api, _repo, edited_type, base, _cache):
        if base == "a":
            raise RuntimeError("下载失败")
        return manifest.SampleMeta(base=base, edited_type="global_style", prompt="p"), Path("o"), Path("e")

    monkeypatch.setattr(watch, "fetch_sample", fake_fetch)
    monkeypatch.setattr("openve_watcher.gemini.evaluate_video_pair", lambda *a, **k: ([5, 5, 5], "raw"))

    added, _, _ = watch.process_repo(api, "r", args, ["k"])
    assert added == 1, "一条失败不应终止整轮"
    assert "下载失败" in capsys.readouterr().err


def test_process_repo_pushes_scores_when_asked(tmp_path, monkeypatch):
    files = ["samples/t/a/meta.json", "samples/t/a/original.mp4", "samples/t/a/edited.mp4"]
    api = FakeApi(files=files)
    args = _args(tmp_path, push_scores=True)
    monkeypatch.setattr(
        watch,
        "fetch_sample",
        lambda *a: (manifest.SampleMeta(base="a", edited_type="global_style", prompt="p"), Path("o"), Path("e")),
    )
    monkeypatch.setattr("openve_watcher.gemini.evaluate_video_pair", lambda *a, **k: ([3, 3, 3], "raw"))
    watch.process_repo(api, "r", args, ["k"])
    assert api.uploaded == [manifest.score_path(args.model)]


def test_append_result_is_line_per_row(tmp_path):
    path = tmp_path / "a" / "s.jsonl"
    watch.append_result(path, {"base": "x", "scores": [1, 2, 3]})
    watch.append_result(path, {"base": "y", "scores": [4, 5, 1]})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert watch.scored_bases(path) == {"x", "y"}
