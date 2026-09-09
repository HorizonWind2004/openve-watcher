"""两侧唯一的契约：每个样本一个自包含目录。

仓库布局（dataset repo）::

    run.json                                  本次推理的运行级信息
    samples/<edited_type>/<base>/meta.json     该样本的类别与编辑指令
    samples/<edited_type>/<base>/original.mp4  原始视频
    samples/<edited_type>/<base>/edited.mp4    模型编辑后的视频
    scores/<model>_gemini_video_score.jsonl    打分侧回传（可选）

刻意不用一个集中的 manifest 文件：那需要读-改-写，两个推送进程或一次
中断就会把它写坏。用"每个样本一个 meta.json"，仓库文件列表本身就是清单，
新增样本永远是纯追加，天然幂等。
"""

import json
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

SAMPLES_ROOT = "samples"
SCORES_ROOT = "scores"
META_NAME = "meta.json"
ORIGINAL_NAME = "original.mp4"
EDITED_NAME = "edited.mp4"
RUN_NAME = "run.json"


@dataclass(frozen=True)
class SampleMeta:
    """打分侧需要的全部信息，除了两个视频文件本身。"""

    base: str
    edited_type: str
    prompt: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "SampleMeta":
        obj = json.loads(text)
        missing = {"base", "edited_type", "prompt"} - set(obj)
        if missing:
            raise ValueError(f"meta.json 缺少字段: {sorted(missing)}")
        return cls(base=obj["base"], edited_type=obj["edited_type"], prompt=obj["prompt"])


def sample_dir(edited_type: str, base: str) -> str:
    return f"{SAMPLES_ROOT}/{edited_type}/{base}"


def sample_paths(edited_type: str, base: str) -> dict:
    root = sample_dir(edited_type, base)
    return {
        "meta": f"{root}/{META_NAME}",
        "original": f"{root}/{ORIGINAL_NAME}",
        "edited": f"{root}/{EDITED_NAME}",
    }


def parse_sample_dir(path: str):
    """从仓库内路径反解出 (edited_type, base)；不是样本路径则返回 None。"""
    parts = PurePosixPath(path).parts
    if len(parts) < 4 or parts[0] != SAMPLES_ROOT:
        return None
    return parts[1], parts[2]


def complete_samples(repo_files) -> dict:
    """从仓库文件列表里挑出三件套齐全的样本。

    只有 meta/original/edited 都在才算完整——上传是逐文件提交的，
    中断会留下半个样本目录，打分侧必须自己识别并跳过。
    """
    seen = {}
    for path in repo_files:
        parsed = parse_sample_dir(path)
        if parsed is None:
            continue
        name = PurePosixPath(path).name
        if name not in (META_NAME, ORIGINAL_NAME, EDITED_NAME):
            continue
        seen.setdefault(parsed, set()).add(name)
    required = {META_NAME, ORIGINAL_NAME, EDITED_NAME}
    return {key: sample_dir(*key) for key, names in seen.items() if required <= names}


def score_path(model: str) -> str:
    """回传打分的仓库内路径。协议写进文件名，避免和抽帧代理的旧分数混用。"""
    safe = model.replace("/", "_")
    return f"{SCORES_ROOT}/{safe}_gemini_video_score.jsonl"
