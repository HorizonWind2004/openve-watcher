"""打分侧：爬 HuggingFace 上的 OpenVE 结果仓库，对新样本跑 Gemini 打分。

这一侧不需要 OpenVE-Bench、不需要 bench CSV、也不需要访问集群。
每个样本目录自带 original.mp4 / edited.mp4 / meta.json，是自包含的。
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import manifest, parse, progress, prompts

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_PREFIX = "openve_"


def preflight_env():
    """关掉那些环境里开了、包却没装的加速开关。

    HF_HUB_ENABLE_HF_TRANSFER=1 而 hf_transfer 未安装时，huggingface_hub 的
    **下载**会直接抛 ValueError，上传却照常——所以推送侧一切正常，
    爬取侧一上手就炸。hf_transfer 只是个加速器，关掉它只是慢一点，
    比让使用者去读一段无关的堆栈要好。返回被关掉的开关名，便于测试。
    """
    disabled = []
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") in ("1", "true", "True"):
        try:
            import hf_transfer  # noqa: F401
        except ImportError:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            disabled.append("HF_HUB_ENABLE_HF_TRANSFER")
            print(
                "[preflight] 环境里开了 HF_HUB_ENABLE_HF_TRANSFER 但没装 hf_transfer，"
                "已关闭（下载会稍慢）。要加速就 pip install hf_transfer。",
                file=sys.stderr,
                flush=True,
            )
    return disabled


def load_api_keys(key_file, env_name="GEMINI_API_KEY"):
    """从文件（每行一个 key）或环境变量读取 key，支持多 key 轮询。"""
    keys = []
    if key_file:
        for line in Path(key_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.append(line)
    env = os.environ.get(env_name, "").strip()
    if env:
        keys.extend(part.strip() for part in env.split(",") if part.strip())
    # 去重但保序，避免同一个 key 被当成两个来轮询。
    seen = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


def discover_repos(api, author, prefix, collection_slug=None):
    """列出候选结果仓库。

    默认按 ``author`` + 名字前缀发现，因为这个不需要人工维护；
    collection 需要有人把每个仓库手动（或用 --collection 推送时）加进去，
    漏加一个就会被静默跳过，所以只作为可选来源。
    """
    if collection_slug:
        from huggingface_hub import get_collection

        collection = get_collection(collection_slug, token=api.token)
        return sorted(
            item.item_id
            for item in collection.items
            if item.item_type == "dataset" and item.item_id.split("/")[-1].startswith(prefix)
        )
    found = set()
    for dataset in api.list_datasets(author=author, search=prefix):
        name = dataset.id.split("/")[-1]
        if name.startswith(prefix):
            found.add(dataset.id)
    return sorted(found)


def scored_bases(jsonl_path):
    """读已有结果，返回已打过分的 base 集合，用来断点续跑。

    只认三项分数齐全且落在 1-5 的行；写一半被截断的行直接忽略，
    这样中断后重跑会自动补上那一条。
    """
    path = Path(jsonl_path)
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        scores = row.get("scores")
        base = row.get("base")
        if not base or not isinstance(scores, list) or len(scores) != 3:
            continue
        if all(isinstance(s, int) and 1 <= s <= 5 for s in scores):
            done.add(base)
    return done


def fetch_sample(api, repo_id, edited_type, base, cache_dir):
    from huggingface_hub import hf_hub_download

    paths = manifest.sample_paths(edited_type, base)
    local = {}
    for key, repo_path in paths.items():
        local[key] = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_path,
            cache_dir=cache_dir,
            token=api.token,
        )
    meta = manifest.SampleMeta.from_json(Path(local["meta"]).read_text(encoding="utf-8"))
    return meta, Path(local["original"]), Path(local["edited"])


def system_prompt(edited_type, edit_prompt, json_mode):
    template = prompts.prompt_type.get(edited_type)
    if template is None:
        raise KeyError(f"没有这个类别的评分提示词: {edited_type}")
    text = template.format(edit_prompt=edit_prompt)
    if json_mode:
        text += parse.JSON_INSTRUCTION
    return text


def score_sample(api, repo_id, edited_type, base, args, keys, key_index):
    from .gemini import evaluate_video_pair

    meta, original, edited = fetch_sample(api, repo_id, edited_type, base, args.cache_dir)
    # 上传端把类别同时写进目录名和 meta.json。两者不一致说明数据坏了，
    # 此时按哪个都可能用错评分标准，所以直接失败而不是猜一个。
    if (meta.edited_type, meta.base) != (edited_type, base):
        raise ValueError(
            f"meta.json 与目录不一致: 目录 {edited_type}/{base}，meta {meta.edited_type}/{meta.base}"
        )
    prompt = system_prompt(meta.edited_type, meta.prompt, args.json_mode)
    scores, raw = evaluate_video_pair(
        original,
        edited,
        prompt,
        api_key=keys[key_index % len(keys)],
        model=args.model,
        timeout=args.timeout,
        parse_response=parse.parse_json if args.json_mode else parse.parse_plain,
    )
    return {
        "base": meta.base,
        "edited_type": meta.edited_type,
        "prompt": meta.prompt,
        "scores": scores,
        "protocol": "gemini_video",
        "model": args.model,
        "repo": repo_id,
        "raw": raw,
    }


def append_result(jsonl_path, row):
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_repo(api, repo_id, args, keys):
    """对一个仓库里所有未打分的完整样本打分，返回 (新增, 仓库总数, 已完成)。"""
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    samples = manifest.complete_samples(files)

    repo_dir = Path(args.out_dir) / repo_id.replace("/", "__")
    out_path = repo_dir / Path(manifest.score_path(args.model)).name
    state = progress.state_path(args.out_dir, repo_id)

    # 两份台账取并集判定已完成：txt 只记做过没做过，jsonl 存完整结果。
    # 先把只在 jsonl 里的补进 txt，覆盖「写完 jsonl 就崩溃」和旧版本升级两种情况。
    from_jsonl = {(t, b) for (t, b) in sorted(samples) if b in scored_bases(out_path)}
    recovered = progress.reconcile(state, repo_id, from_jsonl)
    if recovered:
        print(f"[reconcile] {repo_id}: 从 jsonl 补回 {recovered} 条进度", flush=True)
    done = progress.load(state) | from_jsonl

    pending = [(t, b) for (t, b) in sorted(samples) if (t, b) not in done]
    print(f"[progress] {repo_id}: {progress.summarise(len(samples), len(done), len(pending))}", flush=True)
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        return 0, len(samples), len(done)

    added = 0
    if args.max_workers <= 1:
        for index, (edited_type, base) in enumerate(pending):
            added += _one(api, repo_id, edited_type, base, args, keys, index, out_path, state)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(_one, api, repo_id, t, b, args, keys, i, out_path, state): (t, b)
                for i, (t, b) in enumerate(pending)
            }
            for future in as_completed(futures):
                added += future.result()
    if args.push_scores and added:
        _upload_scores(api, repo_id, out_path, args.model)
    return added, len(samples), len(done)


def _one(api, repo_id, edited_type, base, args, keys, key_index, out_path, state):
    try:
        row = score_sample(api, repo_id, edited_type, base, args, keys, key_index)
    except Exception as exc:  # noqa: BLE001 - 单条失败不该终止整轮
        print(f"[fail] {repo_id} {edited_type}/{base}: {exc}", file=sys.stderr, flush=True)
        return 0
    # 顺序不能反：jsonl 先写。若在两次写入之间崩溃，txt 缺这条但 jsonl 有，
    # 下一轮 reconcile 会补上；反过来会造成「txt 说做过、结果却不存在」。
    append_result(out_path, row)
    progress.append(state, repo_id, edited_type, base)
    print(f"[scored] {repo_id} {edited_type}/{base} -> {row['scores']}", flush=True)
    return 1


def _upload_scores(api, repo_id, out_path, model):
    try:
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=manifest.score_path(model),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="update gemini_video scores",
        )
        print(f"[push-scores] {repo_id} 已回传", flush=True)
    except Exception as exc:  # noqa: BLE001 - 回传失败不该丢掉本地结果
        print(f"[push-scores] 回传失败（本地结果已保存）: {exc}", file=sys.stderr, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="openve-watch",
        description="爬 HuggingFace 上的 OpenVE 推理结果并跑 Gemini 打分（打分侧运行）",
    )
    parser.add_argument("--author", default="sanaka87", help="结果仓库的所有者")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="仓库名前缀")
    parser.add_argument("--collection", default=None, help="改从 collection slug 发现仓库")
    parser.add_argument("--out-dir", default="./openve-scores", help="打分结果落盘目录")
    parser.add_argument("--cache-dir", default=None, help="HF 下载缓存目录")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gemini-key-file", default=None, help="每行一个 key；也读 GEMINI_API_KEY")
    parser.add_argument("--hf-token", default=None, help="HF token；默认取 HF_TOKEN")
    parser.add_argument("--timeout", type=float, default=600.0, help="单条上传+打分的超时（秒）")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--json-mode", action="store_true", help="要求 Gemini 返回 JSON")
    parser.add_argument("--push-scores", action="store_true", help="把分数回传到同一个 HF 仓库")
    parser.add_argument("--limit", type=int, default=0, help="每个仓库每轮最多打多少条（0 = 不限）")
    parser.add_argument("--once", action="store_true", help="只跑一轮就退出")
    parser.add_argument("--interval", type=float, default=300.0, help="轮询间隔（秒）")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    preflight_env()
    from huggingface_hub import HfApi

    keys = load_api_keys(args.gemini_key_file)
    if not keys:
        print("没有 Gemini API key：用 --gemini-key-file 或设 GEMINI_API_KEY", file=sys.stderr)
        return 2
    api = HfApi(token=args.hf_token or os.environ.get("HF_TOKEN"))

    while True:
        repos = discover_repos(api, args.author, args.prefix, args.collection)
        if not repos:
            print(f"[watch] 没找到 {args.author}/{args.prefix}* 的仓库", flush=True)
        for repo_id in repos:
            try:
                added, total, done = process_repo(api, repo_id, args, keys)
            except Exception as exc:  # noqa: BLE001 - 某个仓库出问题不该拖垮其它
                print(f"[error] {repo_id}: {exc}", file=sys.stderr, flush=True)
                continue
            print(f"[watch] {repo_id}: 仓库 {total} 条，已打分 {done}，本轮新增 {added}", flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
