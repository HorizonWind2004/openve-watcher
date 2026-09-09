"""把打分结果提交并推到当前 git 仓库（通常是使用者自己 fork 的那个）。

为什么不用 `--push-scores`：那个是往源 HF dataset 仓库回写，只有产出推理结果
的人有写权限。协作者拿到的是只读的公开数据集，所以他们要发布结果，
唯一顺手的地方是自己 fork 的 GitHub 仓库。
"""

import subprocess
from pathlib import Path


def nearest_existing(path) -> Path:
    """向上找到最近的已存在目录。

    `--out-dir ./scores` 在第一次运行前是不存在的，但它父目录在 fork 的
    clone 里就已经足够说明"能不能推"。直接把不存在的路径交给 subprocess 的
    cwd 会抛 FileNotFoundError，而那与"不是 git 仓库"是两件事。
    """
    path = Path(path).resolve()
    while not path.is_dir():
        if path.parent == path:
            return path
        path = path.parent
    return path


def _git(args, cwd, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=str(nearest_existing(cwd)),
        check=check,
        capture_output=True,
        text=True,
    )


def is_git_repo(path) -> bool:
    result = _git(["rev-parse", "--is-inside-work-tree"], path, check=False)
    return result.returncode == 0 and result.stdout.strip() == "true"


def current_branch(path) -> str:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], path).stdout.strip()


def publish(out_dir, *, repo_root=None, remote="origin", message=None, push=True):
    """把 out_dir 下的改动提交并推送。返回一句人类可读的结果说明。

    没有改动就什么都不做——挂着轮询时大多数轮次都是没有改动的，
    不该产生一串空提交。
    """
    out_dir = Path(out_dir).resolve()
    root = Path(repo_root).resolve() if repo_root else out_dir
    if not is_git_repo(root):
        return f"跳过发布：{root} 不是 git 仓库"

    try:
        rel = out_dir.relative_to(_git(["rev-parse", "--show-toplevel"], root).stdout.strip())
    except ValueError:
        rel = out_dir

    _git(["add", "--", str(rel)], root)
    staged = _git(["diff", "--cached", "--name-only"], root).stdout.strip()
    if not staged:
        return "跳过发布：没有新的分数"

    count = len(staged.splitlines())
    _git(["commit", "-m", message or f"update openve scores ({count} files)"], root)
    if not push:
        return f"已提交 {count} 个文件（未推送）"

    branch = current_branch(root)
    result = _git(["push", remote, f"HEAD:{branch}"], root, check=False)
    if result.returncode != 0:
        # 推送失败要说清楚，但本地提交已经在了，不该当成致命错误。
        return f"已提交 {count} 个文件，但推送到 {remote}/{branch} 失败：{result.stderr.strip().splitlines()[-1:]}"
    return f"已提交并推送 {count} 个文件到 {remote}/{branch}"
