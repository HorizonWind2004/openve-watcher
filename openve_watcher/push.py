"""集群侧：把已完成的 OpenVE 推理样本推到 HuggingFace dataset 仓库。

推理进程一边跑一边落盘，本模块只做"发现完整样本 → 组装自包含目录 → 上传"。
不改动、不删除推理输出。
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import manifest

INFO_SUFFIX = "_info.txt"
DERIVED_SUFFIXES = ("_compare.mp4", "_reason_edit.mp4")


def load_bench(bench_csv, bench_dir):
    """读 benchmark_videos.csv，返回 base -> (edited_type, prompt, 原始视频路径)。

    CSV 里的 original_video 可能写成 ``OpenVE-Bench/xxx.mp4``，也可能是相对路径，
    两种都归一到 bench_dir 下。
    """
    bench_dir = Path(bench_dir)
    table = {}
    with open(bench_csv, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            original = row["original_video"]
            if original.startswith("OpenVE-Bench/"):
                path = bench_dir / original[len("OpenVE-Bench/") :]
            else:
                path = Path(original)
                if not path.is_absolute():
                    path = bench_dir / path
            table[Path(original).stem] = (row["edited_type"], row["prompt"], path)
    return table


def edited_video_for(case_dir, base):
    """挑出主视频。

    ``_compare.mp4`` 是三联对比图、``_reason_edit.mp4`` 多一帧推理帧，
    都不是要打分的对象，必须排掉。
    """
    preferred = case_dir / f"gen_{base}.mp4"
    if preferred.exists():
        return preferred
    for path in sorted(case_dir.glob("gen_*.mp4")):
        if path.name.endswith(DERIVED_SUFFIXES):
            continue
        return path
    return None


def discover(inference_dir, bench):
    """扫描推理输出目录，产出可上传的样本。

    完整的判据是主视频与 ``*_info.txt`` 同时存在——info.txt 是推理侧最后写的
    文件，只有它落盘才说明这条样本真的做完了。
    """
    inference_dir = Path(inference_dir)
    found = []
    for info in sorted(inference_dir.glob(f"*/*/*{INFO_SUFFIX}")):
        case_dir = info.parent
        base = case_dir.name
        entry = bench.get(base)
        if entry is None:
            continue
        edited_type, prompt, original = entry
        edited = edited_video_for(case_dir, base)
        if edited is None or not original.exists():
            continue
        found.append((edited_type, base, prompt, original, edited))
    return found


def already_uploaded(api, repo_id):
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return set()
    return set(manifest.complete_samples(files))


def stage_sample(tmp, base, edited_type, prompt, original, edited):
    root = Path(tmp) / base
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, root / manifest.ORIGINAL_NAME)
    shutil.copy2(edited, root / manifest.EDITED_NAME)
    meta = manifest.SampleMeta(base=base, edited_type=edited_type, prompt=prompt)
    (root / manifest.META_NAME).write_text(meta.to_json(), encoding="utf-8")
    return root


# HuggingFace 限制每个仓库每小时 256 次 commit。一个样本一次 commit 时，
# 一个 388 条的 run 需要 388 次，必然撞限；补传积压更是瞬间突发。
# 所以默认按批提交：一批仍是一次 commit，批内每个样本要么整条可见、
# 要么完全不可见，"看不到半个样本"这个性质不变。
COMMITS_PER_HOUR = 256
DEFAULT_BATCH_SIZE = 25


def _is_rate_limited(exc) -> bool:
    return "429" in str(exc) or "Too Many Requests" in str(exc)


def _commit_with_backoff(action, *, attempts=4, rate_limit_wait=600.0):
    """提交并重试。

    限流和别的错误要分开对待：限流要等到窗口滚动（分钟级），
    短促重试只会白白烧掉剩下的尝试次数。
    """
    last = None
    for attempt in range(attempts):
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - 重试所有可恢复错误
            last = exc
            if attempt == attempts - 1:
                break
            if _is_rate_limited(exc):
                print(
                    f"[rate-limit] HF 每小时 {COMMITS_PER_HOUR} 次 commit 已用尽，"
                    f"等 {rate_limit_wait / 60:.0f} 分钟后重试",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(rate_limit_wait)
            else:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"HF 提交连续 {attempts} 次失败: {last}") from last


def push_once(api, repo_id, samples, *, dry_run=False, limit=0, batch_size=DEFAULT_BATCH_SIZE):
    """上传尚未在仓库里的样本，返回本次上传的数量。"""
    have = already_uploaded(api, repo_id)
    pending = [s for s in samples if (s[0], s[1]) not in have]
    if limit:
        pending = pending[:limit]
    if not pending:
        return 0
    if dry_run:
        for edited_type, base, *_ in pending:
            print(f"[dry-run] 待上传 {edited_type}/{base}", flush=True)
        batches = (len(pending) + batch_size - 1) // batch_size
        print(f"[dry-run] 将分 {batches} 次 commit（每批 {batch_size} 条）", flush=True)
        return len(pending)

    uploaded = 0
    batch_size = max(1, batch_size)
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / manifest.SAMPLES_ROOT
            for edited_type, base, prompt, original, edited in chunk:
                staged = root / edited_type
                staged.mkdir(parents=True, exist_ok=True)
                stage_sample(staged, base, edited_type, prompt, original, edited)
            names = ", ".join(f"{t}/{b}" for t, b, *_ in chunk[:2])
            more = f" 等 {len(chunk)} 条" if len(chunk) > 2 else ""
            _commit_with_backoff(
                lambda: api.upload_folder(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=str(root),
                    path_in_repo=manifest.SAMPLES_ROOT,
                    commit_message=f"add {len(chunk)} samples ({names}{more})",
                )
            )
        uploaded += len(chunk)
        print(f"[uploaded] {len(chunk)} 条（累计 {uploaded}/{len(pending)}）：{names}{more}", flush=True)
    return uploaded


def write_run_json(api, repo_id, payload, *, dry_run=False):
    if dry_run:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / manifest.RUN_NAME
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _commit_with_backoff(
            lambda: api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=manifest.RUN_NAME,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="update run.json",
            )
        )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="openve-push",
        description="把已完成的 OpenVE 推理样本推到 HuggingFace（集群侧运行）",
    )
    parser.add_argument("--inference-dir", required=True, help="形如 runs/inference/openvebench/<run>")
    parser.add_argument("--repo", required=True, help="目标 dataset 仓库，例如 sanaka87/openve_33f_fa_cfg1")
    parser.add_argument(
        "--bench-csv", default=os.path.expandvars("${I2V_CACHE_ROOT}/OpenVE-Bench/benchmark_videos.csv")
    )
    parser.add_argument("--bench-dir", default=os.path.expandvars("${I2V_CACHE_ROOT}/OpenVE-Bench"))
    parser.add_argument("--token", default=None, help="HF token；默认取 HF_TOKEN 或已登录凭据")
    parser.add_argument("--private", action="store_true", help="建成私有仓库（默认公开）")
    parser.add_argument("--collection", default=None, help="可选：同时加入该 collection（slug）")
    parser.add_argument("--watch", action="store_true", help="持续监视，推理边跑边传")
    parser.add_argument("--interval", type=float, default=60.0, help="--watch 的轮询间隔（秒）")
    parser.add_argument("--limit", type=int, default=0, help="每轮最多上传多少条（0 = 不限）")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"每次 commit 打包多少条样本（HF 上限 {COMMITS_PER_HOUR} commit/小时/仓库）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印要传什么，不真的传")
    parser.add_argument("--note", default=None, help="写进 run.json 的备注")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from huggingface_hub import HfApi

    api = HfApi(token=args.token or os.environ.get("HF_TOKEN"))
    bench = load_bench(args.bench_csv, args.bench_dir)
    if not bench:
        print(f"bench CSV 里没有任何行: {args.bench_csv}", file=sys.stderr)
        return 2

    if not args.dry_run:
        api.create_repo(repo_id=args.repo, repo_type="dataset", private=args.private, exist_ok=True)
        write_run_json(
            api,
            args.repo,
            {
                "inference_dir": str(Path(args.inference_dir).resolve()),
                "bench_csv": str(Path(args.bench_csv).resolve()),
                "note": args.note,
                "expected_samples": len(bench),
            },
        )
        if args.collection:
            _add_to_collection(api, args.collection, args.repo)

    while True:
        samples = discover(args.inference_dir, bench)
        count = push_once(
            api, args.repo, samples, dry_run=args.dry_run, limit=args.limit, batch_size=args.batch_size
        )
        print(f"[push] 本地完整 {len(samples)} 条，本轮上传 {count} 条", flush=True)
        if not args.watch:
            return 0
        time.sleep(args.interval)


def _add_to_collection(api, slug, repo_id):
    from huggingface_hub import add_collection_item

    try:
        add_collection_item(
            collection_slug=slug,
            item_id=repo_id,
            item_type="dataset",
            token=api.token,
            exists_ok=True,
        )
        print(f"[collection] {repo_id} 已加入 {slug}", flush=True)
    except Exception as exc:  # noqa: BLE001 - collection 失败不该阻塞上传
        print(f"[collection] 加入失败（不影响上传）: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
