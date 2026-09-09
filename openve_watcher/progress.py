"""本地进度台账：一行一个已打分样本的纯文本文件。

为什么在 jsonl 之外还要一个 txt：jsonl 里存着完整结果（含 Gemini 原始回复），
文件大、解析慢，而且一旦某行写坏，那条样本的"做过没做过"就说不清了。
这个 txt 只回答一个问题——**这条样本打过分了吗**——所以它可以做到
只追加、逐行 fsync、坏了一行也只影响那一行。

两个文件互为保险，取**并集**判定「已完成」：

- 防重复：任一文件说做过，就跳过。
- 防漏做：写入顺序固定为「先 jsonl，后 txt」。若在两次写入之间崩溃，
  txt 缺这一条但 jsonl 有，并集仍判定为已完成，不会重复调用 Gemini；
  反过来永远不会发生，所以不存在「txt 说做过、其实没做」的情况。
"""

import os
from pathlib import Path

STATE_NAME = "progress.txt"
SEPARATOR = "\t"


def state_path(out_dir, repo_id) -> Path:
    return Path(out_dir) / repo_id.replace("/", "__") / STATE_NAME


def _key(edited_type, base) -> str:
    return f"{edited_type}/{base}"


def load(path) -> set:
    """读回已完成的 (edited_type, base)。文件不存在视为空。

    格式不合的行直接忽略而不是报错：这个文件是崩溃现场的产物，
    半行、空行都可能出现，为了一行坏数据放弃整个进度表是不值得的。
    """
    path = Path(path)
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(SEPARATOR)
        if len(parts) < 2:
            continue
        key = parts[1].strip()
        if key.count("/") != 1:
            continue
        edited_type, base = key.split("/")
        if edited_type and base:
            done.add((edited_type, base))
    return done


def append(path, repo_id, edited_type, base) -> None:
    """追加一条并落盘。

    fsync 是必要的：这个文件的全部价值在于崩溃后还准确，
    留在页缓存里没写下去的进度和没有进度是一回事。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{repo_id}{SEPARATOR}{_key(edited_type, base)}\n")
        handle.flush()
        os.fsync(handle.fileno())


def reconcile(path, repo_id, scored) -> int:
    """把只在 jsonl 里、txt 里没有的条目补进 txt，返回补了多少条。

    用于两种情况：从只有 jsonl 的旧版本升级上来，以及在
    「写完 jsonl、还没写 txt」的瞬间崩溃过。
    """
    have = load(path)
    missing = sorted(set(scored) - have)
    for edited_type, base in missing:
        append(path, repo_id, edited_type, base)
    return len(missing)


def summarise(repo_total, done, pending) -> str:
    """一行进度摘要。漏做的条数是 repo 总数减去已完成，明确打出来。"""
    return f"仓库 {repo_total} 条 / 已打分 {done} 条 / 待打分 {pending} 条"
