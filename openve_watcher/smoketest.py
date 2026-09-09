"""一次验完整条链路，包含一次真实的 Gemini 调用。

存在的理由：把配置交给 agent 或者交给很忙的人时，最坏的失败不是报错，
而是"看起来跑起来了、其实一条分都没打出来"。这个命令给出明确的成功判据——
每一步 PASS/FAIL，失败时直接说该改什么，最后退出码非零。
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

from . import manifest, parse, progress, prompts, publish, watch

# OpenVE-Bench 七类计划实际用到的类别。
PLANNED_TYPES = (
    "global_style",
    "local_change",
    "background_change",
    "local_remove",
    "local_add",
    "creative_edit",
    "subtitle_edit",
)


class Reporter:
    def __init__(self):
        self.failed = []
        self.step = 0

    def ok(self, name, detail=""):
        self.step += 1
        print(f"  [{self.step}] PASS  {name}" + (f" — {detail}" if detail else ""), flush=True)

    def fail(self, name, detail, fix):
        self.step += 1
        self.failed.append(name)
        print(f"  [{self.step}] FAIL  {name} — {detail}", flush=True)
        print(f"            怎么改：{fix}", flush=True)

    def skip(self, name, why):
        self.step += 1
        print(f"  [{self.step}] SKIP  {name} — {why}", flush=True)


def check_dependencies(report, need_gemini):
    try:
        import huggingface_hub

        report.ok("huggingface_hub 已安装", huggingface_hub.__version__)
    except ImportError as exc:
        report.fail("huggingface_hub 已安装", str(exc), "pip install -e .")
        return False
    if not need_gemini:
        report.skip("google-genai 已安装", "--no-gemini 跳过打分")
        return True
    try:
        from google import genai  # noqa: F401

        report.ok("google-genai 已安装")
    except ImportError as exc:
        report.fail("google-genai 已安装", str(exc), "pip install -e '.[score]'")
        return False
    return True


def check_prompts(report):
    missing = [t for t in PLANNED_TYPES if t not in prompts.prompt_type]
    if missing:
        report.fail("七类评分提示词齐全", f"缺 {missing}", "别改 prompts.py；重新 clone")
        return False
    try:
        for edited_type in PLANNED_TYPES:
            text = watch.system_prompt(edited_type, "冒烟测试用的编辑指令", json_mode=False)
            assert "冒烟测试用的编辑指令" in text and "{edit_prompt}" not in text
    except Exception as exc:  # noqa: BLE001
        report.fail("提示词能正确注入编辑指令", str(exc), "别改 prompts.py；重新 clone")
        return False
    leaked = sorted(set(prompts.NON_OFFICIAL_PROMPT_TYPE) & set(prompts.prompt_type))
    if leaked:
        report.fail("非官方提示词已隔离", f"{leaked} 混进了 prompt_type", "别改 prompts.py；重新 clone")
        return False
    report.ok("提示词对齐官方", f"{len(PLANNED_TYPES)} 类可用，非官方类别已隔离")
    return True


def check_discovery(report, api, author, prefix):
    try:
        repos = watch.discover_repos(api, author, prefix)
    except Exception as exc:  # noqa: BLE001
        report.fail(
            "能发现 HF 上的结果仓库",
            str(exc),
            "检查网络是否能访问 huggingface.co（公开数据集不需要 token）",
        )
        return None
    if not repos:
        report.fail(
            "能发现 HF 上的结果仓库",
            f"{author}/{prefix}* 一个都没找到",
            f"确认 --author（现在是 {author}）和 --prefix（现在是 {prefix}）没写错",
        )
        return None
    report.ok("能发现 HF 上的结果仓库", f"{len(repos)} 个：{', '.join(repos)}")
    return repos


def check_sample(report, api, repo_id, cache_dir):
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        report.fail("能列出仓库文件", str(exc), "检查网络与仓库是否公开")
        return None
    samples = manifest.complete_samples(files)
    if not samples:
        report.fail(
            "仓库里有完整样本",
            f"{repo_id} 里没有三件套齐全的样本",
            "等推理侧上传，或换一个仓库试",
        )
        return None
    report.ok("仓库里有完整样本", f"{repo_id}：{len(samples)} 条")

    edited_type, base = sorted(samples)[0]
    try:
        meta, original, edited = watch.fetch_sample(api, repo_id, edited_type, base, cache_dir)
    except Exception as exc:  # noqa: BLE001
        report.fail(
            "能下载一个样本",
            str(exc),
            "若报 hf_transfer 相关错误，pip install hf_transfer 或 unset HF_HUB_ENABLE_HF_TRANSFER",
        )
        return None
    sizes = f"original {original.stat().st_size // 1024} KB / edited {edited.stat().st_size // 1024} KB"
    report.ok("能下载一个样本", f"{edited_type}/{base}，{sizes}")

    if (meta.edited_type, meta.base) != (edited_type, base):
        report.fail(
            "meta.json 与目录一致",
            f"目录 {edited_type}/{base} vs meta {meta.edited_type}/{meta.base}",
            "上传侧的数据坏了，联系产出推理结果的人",
        )
        return None
    report.ok("meta.json 与目录一致", f"类别 {meta.edited_type}")
    return repo_id, edited_type, base, meta, original, edited


def check_gemini(report, args, keys, sample):
    from .gemini import evaluate_video_pair

    repo_id, edited_type, base, meta, original, edited = sample
    prompt = watch.system_prompt(meta.edited_type, meta.prompt, args.json_mode)
    try:
        scores, raw = evaluate_video_pair(
            original,
            edited,
            prompt,
            api_key=keys[0],
            model=args.model,
            timeout=args.timeout,
            parse_response=parse.parse_json if args.json_mode else parse.parse_plain,
        )
    except Exception as exc:  # noqa: BLE001
        report.fail(
            f"能用 {args.model} 真打出一条分数",
            f"{type(exc).__name__}: {exc}",
            "确认 API key 有效、有该模型的权限、且网络能访问 Gemini；"
            f"官方用的模型是 gemini-2.5-pro（当前 --model {args.model}）",
        )
        return None
    report.ok(
        f"能用 {args.model} 真打出一条分数",
        f"指令遵从 {scores[0]} / 画质稳定性 {scores[1]} / 一致性 {scores[2]}",
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


def check_ledger(report, args, repo_id, edited_type, base, row):
    out_path = Path(args.out_dir) / repo_id.replace("/", "__") / Path(manifest.score_path(args.model)).name
    state = progress.state_path(args.out_dir, repo_id)
    try:
        watch.append_result(out_path, row)
        progress.append(state, repo_id, edited_type, base)
    except Exception as exc:  # noqa: BLE001
        report.fail("结果与台账能落盘", str(exc), f"检查 --out-dir（{args.out_dir}）是否可写")
        return False
    if (edited_type, base) not in progress.load(state):
        report.fail("台账能读回", "刚写的条目读不回来", "检查磁盘是否写满")
        return False
    if base not in watch.scored_bases(out_path):
        report.fail("jsonl 能读回", "刚写的结果读不回来", "检查磁盘是否写满")
        return False
    report.ok("结果与台账能落盘并读回", f"{out_path}")
    report.ok("重跑不会重复打分", "台账已记录，下一轮会跳过这条")
    return True


def check_git(report, args):
    if not args.git_publish:
        report.skip("能推到自己 fork", "没加 --git-publish")
        return True
    root = Path(args.out_dir)
    if not publish.is_git_repo(root):
        near = publish.nearest_existing(root)
        report.fail(
            "能推到自己 fork",
            f"{root.resolve()} 不在 git 仓库里（往上找到的最近目录是 {near}）",
            "在 fork 的 clone 里运行，并把 --out-dir 指到 clone 内部，例如 --out-dir ./scores",
        )
        return False
    branch = publish.current_branch(root)
    report.ok("能推到自己 fork", f"分支 {branch}（真实推送在 openve-watch 每轮结束时进行）")
    return True


def build_parser():
    parser = argparse.ArgumentParser(
        prog="openve-smoketest",
        description="一次验完整条链路（含一次真实 Gemini 调用），给出明确的成功判据",
    )
    parser.add_argument("--author", default="sanaka87")
    parser.add_argument("--prefix", default=watch.DEFAULT_PREFIX)
    parser.add_argument("--out-dir", default="./openve-scores")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model", default=watch.DEFAULT_MODEL)
    parser.add_argument("--gemini-key-file", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--git-publish", action="store_true", help="同时检查能否推到自己 fork")
    parser.add_argument("--no-gemini", action="store_true", help="跳过真实打分，只验 HF 侧")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print("openve-watcher 冒烟测试", flush=True)
    print(f"  作者={args.author} 前缀={args.prefix} 模型={args.model} 输出={args.out_dir}", flush=True)
    if args.model != watch.DEFAULT_MODEL:
        print(f"  注意：官方用的是 {watch.DEFAULT_MODEL}，换模型后分数不可与官方口径比较", flush=True)
    print(flush=True)

    report = Reporter()
    need_gemini = not args.no_gemini
    watch.preflight_env()

    if not check_dependencies(report, need_gemini):
        return _summary(report)
    check_prompts(report)

    keys = watch.load_api_keys(args.gemini_key_file)
    if need_gemini and not keys:
        report.fail(
            "有 Gemini API key",
            "既没有 --gemini-key-file 也没有 GEMINI_API_KEY",
            "export GEMINI_API_KEY=xxx，或 --gemini-key-file 指向每行一个 key 的文件",
        )
        return _summary(report)
    if need_gemini:
        report.ok("有 Gemini API key", f"{len(keys)} 个")
    else:
        report.skip("有 Gemini API key", "--no-gemini")

    from huggingface_hub import HfApi

    api = HfApi(token=args.hf_token or os.environ.get("HF_TOKEN"))
    repos = check_discovery(report, api, args.author, args.prefix)
    if not repos:
        return _summary(report)

    sample = None
    for repo_id in repos:
        sample = check_sample(report, api, repo_id, args.cache_dir)
        if sample:
            break
    if not sample:
        return _summary(report)

    if need_gemini:
        row = check_gemini(report, args, keys, sample)
        if row:
            check_ledger(report, args, sample[0], sample[1], sample[2], row)
    else:
        report.skip("能真打出一条分数", "--no-gemini")
        report.skip("结果与台账能落盘", "没有结果可写")

    check_git(report, args)
    return _summary(report)


def _summary(report):
    print(flush=True)
    if report.failed:
        print(f"结果：{len(report.failed)} 项失败 — {', '.join(report.failed)}", flush=True)
        print("按上面每项的「怎么改」处理完再重跑本命令。", flush=True)
        return 1
    print("结果：全部通过。可以挂 openve-watch 了。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:  # noqa: BLE001 - 冒烟测试不该以裸堆栈收尾
        traceback.print_exc()
        print("\n结果：冒烟测试自身异常退出，请把上面的堆栈发给 HorizonWind。", flush=True)
        raise SystemExit(2)
