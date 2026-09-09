"""进度台账：防重复、防漏做。"""

import json
from pathlib import Path
from types import SimpleNamespace

from openve_watcher import manifest, progress, watch


def test_append_then_load_round_trip(tmp_path):
    path = tmp_path / progress.STATE_NAME
    progress.append(path, "sanaka87/openve_x", "global_style", "0000_a")
    progress.append(path, "sanaka87/openve_x", "local_add", "0001_b")
    assert progress.load(path) == {("global_style", "0000_a"), ("local_add", "0001_b")}


def test_load_missing_file_is_empty(tmp_path):
    assert progress.load(tmp_path / "nope.txt") == set()


def test_load_tolerates_garbage_lines(tmp_path):
    """崩溃现场会留下半行和空行，坏一行不该毁掉整张进度表。"""
    path = tmp_path / progress.STATE_NAME
    path.write_text(
        "sanaka87/openve_x\tglobal_style/ok\n"
        "\n"
        "# 注释\n"
        "没有分隔符的一行\n"
        "sanaka87/openve_x\t斜杠太多/a/b\n"
        "sanaka87/openve_x\t/缺类别\n"
        "sanaka87/openve_x\tlocal_add/也算\n",
        encoding="utf-8",
    )
    assert progress.load(path) == {("global_style", "ok"), ("local_add", "也算")}


def test_append_is_line_oriented_and_readable(tmp_path):
    path = tmp_path / progress.STATE_NAME
    progress.append(path, "r", "t", "b")
    text = path.read_text(encoding="utf-8")
    assert text == "r\tt/b\n", "格式要简单到能用 wc -l 数进度"


def test_reconcile_backfills_from_jsonl_only_entries(tmp_path):
    path = tmp_path / progress.STATE_NAME
    progress.append(path, "r", "t", "already")
    added = progress.reconcile(path, "r", {("t", "already"), ("t", "from_jsonl")})
    assert added == 1
    assert progress.load(path) == {("t", "already"), ("t", "from_jsonl")}


def test_reconcile_is_idempotent(tmp_path):
    path = tmp_path / progress.STATE_NAME
    scored = {("t", "a"), ("t", "b")}
    assert progress.reconcile(path, "r", scored) == 2
    assert progress.reconcile(path, "r", scored) == 0


class FakeApi:
    def __init__(self, files=()):
        self._files = list(files)
        self.token = None
        self.uploaded = []

    def list_repo_files(self, repo_id, repo_type):
        return list(self._files)

    def upload_file(self, **kwargs):
        self.uploaded.append(kwargs["path_in_repo"])


THREE = [
    "samples/global_style/a/meta.json",
    "samples/global_style/a/original.mp4",
    "samples/global_style/a/edited.mp4",
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


def _stub(monkeypatch, calls):
    monkeypatch.setattr(
        watch,
        "fetch_sample",
        lambda *a: (manifest.SampleMeta(base=a[3], edited_type=a[2], prompt="p"), Path("o"), Path("e")),
    )

    def fake_eval(*args, **kwargs):
        calls.append(1)
        return [4, 4, 4], "raw"

    monkeypatch.setattr("openve_watcher.gemini.evaluate_video_pair", fake_eval)


def test_txt_written_alongside_jsonl(tmp_path, monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    args = _args(tmp_path)
    watch.process_repo(FakeApi(THREE), "sanaka87/openve_x", args, ["k"])
    state = progress.state_path(args.out_dir, "sanaka87/openve_x")
    assert progress.load(state) == {("global_style", "a")}


def test_txt_alone_prevents_a_second_gemini_call(tmp_path, monkeypatch):
    """删掉 jsonl，只留 txt，也不能重复打分——这是防重复的关键。"""
    calls = []
    _stub(monkeypatch, calls)
    args = _args(tmp_path)
    api = FakeApi(THREE)
    watch.process_repo(api, "r", args, ["k"])
    assert len(calls) == 1

    out_path = Path(args.out_dir) / "r" / Path(manifest.score_path(args.model)).name
    out_path.unlink()

    added, _, done = watch.process_repo(api, "r", args, ["k"])
    assert added == 0 and done == 1
    assert len(calls) == 1, "txt 已记录，不该再调用 Gemini"


def test_jsonl_alone_backfills_txt_and_skips(tmp_path, monkeypatch):
    """模拟「写完 jsonl 就崩溃」：txt 缺这条，并集仍判已完成并补回 txt。"""
    calls = []
    _stub(monkeypatch, calls)
    args = _args(tmp_path)
    out_path = Path(args.out_dir) / "r" / Path(manifest.score_path(args.model)).name
    out_path.parent.mkdir(parents=True)
    out_path.write_text(json.dumps({"base": "a", "scores": [3, 3, 3]}) + "\n", encoding="utf-8")

    added, _, done = watch.process_repo(FakeApi(THREE), "r", args, ["k"])
    assert added == 0 and done == 1
    assert calls == [], "并集判定应当跳过，不重复调用 Gemini"
    assert progress.load(progress.state_path(args.out_dir, "r")) == {("global_style", "a")}


def test_failed_sample_is_not_recorded_so_it_retries(tmp_path, monkeypatch):
    """防漏做：失败的样本不能进台账，否则永远不会被重试。"""
    monkeypatch.setattr(
        watch,
        "fetch_sample",
        lambda *a: (manifest.SampleMeta(base="a", edited_type="global_style", prompt="p"), Path("o"), Path("e")),
    )
    monkeypatch.setattr(
        "openve_watcher.gemini.evaluate_video_pair",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Gemini 挂了")),
    )
    args = _args(tmp_path)
    added, _, _ = watch.process_repo(FakeApi(THREE), "r", args, ["k"])
    assert added == 0
    assert progress.load(progress.state_path(args.out_dir, "r")) == set()


def test_pending_count_reveals_missed_samples(tmp_path, monkeypatch, capsys):
    """防漏做：待打分条数会被打出来，仓库有多少、做了多少一目了然。"""
    calls = []
    _stub(monkeypatch, calls)
    files = THREE + [
        "samples/local_add/b/meta.json",
        "samples/local_add/b/original.mp4",
        "samples/local_add/b/edited.mp4",
    ]
    args = _args(tmp_path, limit=1)
    watch.process_repo(FakeApi(files), "r", args, ["k"])
    out = capsys.readouterr().out
    assert "仓库 2 条 / 已打分 0 条 / 待打分 2 条" in out
