import csv
from pathlib import Path

import pytest

from openve_watcher import manifest, push

BENCH_ROWS = [
    {"edited_type": "global_style", "prompt": "印象派化", "original_video": "OpenVE-Bench/videos/0000_a.mp4"},
    {"edited_type": "local_add", "prompt": "加一只猫", "original_video": "videos/0001_b.mp4"},
]


@pytest.fixture
def bench(tmp_path):
    """造一个最小 OpenVE-Bench：CSV 两种路径写法各一行，外加原始视频文件。"""
    bench_dir = tmp_path / "OpenVE-Bench"
    (bench_dir / "videos").mkdir(parents=True)
    for row in BENCH_ROWS:
        (bench_dir / "videos" / Path(row["original_video"]).name).write_bytes(b"orig")
    csv_path = bench_dir / "benchmark_videos.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["edited_type", "prompt", "original_video"])
        writer.writeheader()
        writer.writerows(BENCH_ROWS)
    return csv_path, bench_dir


def make_case(inference_dir, edited_type, base, *, info=True, derived=True):
    case = inference_dir / edited_type / base
    case.mkdir(parents=True)
    (case / f"gen_{base}.mp4").write_bytes(b"edited")
    if derived:
        (case / f"gen_{base}_compare.mp4").write_bytes(b"cmp")
        (case / f"gen_{base}_reason_edit.mp4").write_bytes(b"reason")
    if info:
        (case / f"gen_{base}_info.txt").write_text("{}", encoding="utf-8")
    return case


def test_load_bench_normalises_both_path_forms(bench):
    csv_path, bench_dir = bench
    table = push.load_bench(csv_path, bench_dir)
    assert set(table) == {"0000_a", "0001_b"}
    for base in table:
        assert table[base][2].exists(), f"{base} 的原始视频路径没归一到 bench_dir"


def test_edited_video_excludes_derived_outputs(tmp_path):
    case = make_case(tmp_path, "global_style", "0000_a")
    picked = push.edited_video_for(case, "0000_a")
    assert picked.name == "gen_0000_a.mp4"


def test_edited_video_falls_back_but_still_skips_derived(tmp_path):
    case = tmp_path / "t" / "b"
    case.mkdir(parents=True)
    (case / "gen_b_compare.mp4").write_bytes(b"cmp")
    (case / "gen_b_reason_edit.mp4").write_bytes(b"r")
    assert push.edited_video_for(case, "b") is None, "只有派生文件时不能误当主视频"


def test_discover_requires_info_txt(bench, tmp_path):
    csv_path, bench_dir = bench
    table = push.load_bench(csv_path, bench_dir)
    inference = tmp_path / "inf"
    make_case(inference, "global_style", "0000_a", info=True)
    make_case(inference, "local_add", "0001_b", info=False)
    found = push.discover(inference, table)
    assert [(t, b) for t, b, *_ in found] == [("global_style", "0000_a")]


def test_discover_ignores_cases_absent_from_bench(bench, tmp_path):
    csv_path, bench_dir = bench
    table = push.load_bench(csv_path, bench_dir)
    inference = tmp_path / "inf"
    make_case(inference, "global_style", "9999_unknown")
    assert push.discover(inference, table) == []


class FakeApi:
    """只实现 push 用到的三个方法，记录调用以便断言。"""

    def __init__(self, existing=()):
        self.files = list(existing)
        self.uploads = []
        self.token = None

    def list_repo_files(self, repo_id, repo_type):
        return list(self.files)

    def upload_folder(self, *, repo_id, repo_type, folder_path, path_in_repo, commit_message):
        self.uploads.append(path_in_repo)
        for child in sorted(Path(folder_path).iterdir()):
            self.files.append(f"{path_in_repo}/{child.name}")


def test_push_once_uploads_self_contained_sample(bench, tmp_path):
    csv_path, bench_dir = bench
    table = push.load_bench(csv_path, bench_dir)
    inference = tmp_path / "inf"
    make_case(inference, "global_style", "0000_a")
    api = FakeApi()

    count = push.push_once(api, "sanaka87/openve_test", push.discover(inference, table))

    assert count == 1
    assert api.uploads == ["samples/global_style/0000_a"]
    names = {Path(f).name for f in api.files}
    assert names == {manifest.META_NAME, manifest.ORIGINAL_NAME, manifest.EDITED_NAME}


def test_push_once_is_idempotent(bench, tmp_path):
    csv_path, bench_dir = bench
    table = push.load_bench(csv_path, bench_dir)
    inference = tmp_path / "inf"
    make_case(inference, "global_style", "0000_a")
    api = FakeApi()
    samples = push.discover(inference, table)

    assert push.push_once(api, "r", samples) == 1
    assert push.push_once(api, "r", samples) == 0, "已上传的样本不该重传"
    assert len(api.uploads) == 1


def test_push_once_dry_run_uploads_nothing(bench, tmp_path):
    csv_path, bench_dir = bench
    table = push.load_bench(csv_path, bench_dir)
    inference = tmp_path / "inf"
    make_case(inference, "global_style", "0000_a")
    api = FakeApi()
    assert push.push_once(api, "r", push.discover(inference, table), dry_run=True) == 1
    assert api.uploads == []


def test_push_once_respects_limit(bench, tmp_path):
    csv_path, bench_dir = bench
    table = push.load_bench(csv_path, bench_dir)
    inference = tmp_path / "inf"
    make_case(inference, "global_style", "0000_a")
    make_case(inference, "local_add", "0001_b")
    api = FakeApi()
    assert push.push_once(api, "r", push.discover(inference, table), limit=1) == 1
    assert len(api.uploads) == 1


def test_stage_sample_writes_meta_with_prompt(tmp_path, bench):
    csv_path, bench_dir = bench
    original = bench_dir / "videos" / "0000_a.mp4"
    edited = tmp_path / "e.mp4"
    edited.write_bytes(b"e")
    root = push.stage_sample(tmp_path / "stage", "0000_a", "global_style", "印象派化", original, edited)
    meta = manifest.SampleMeta.from_json((root / manifest.META_NAME).read_text(encoding="utf-8"))
    assert meta.prompt == "印象派化" and meta.edited_type == "global_style"
