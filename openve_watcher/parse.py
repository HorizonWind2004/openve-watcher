"""解析 Gemini 返回的三项评分。

两个函数逐字对应 I2V-transfer ``score.py`` 里的 ``check_format`` 与
``check_json_format``，保持同样的宽松度：失败返回 False 而不是抛异常，
因为调用方要靠这个来触发重试。
"""

import json
import re

SCORE_KEYS = ("instruction_compliance", "visual_quality_stability", "consistency_detail_fidelity")


def parse_plain(out):
    """解析 "Instruction Compliance: 4" 这种逐行文本格式。"""
    try:
        found = {}
        labels = {
            "Instruction Compliance:": "instruct",
            "Visual Quality & Stability:": "vis",
            "Consistency & Detail Fidelity:": "cons",
        }
        for line in (out or "").splitlines():
            line = line.strip()
            for label, key in labels.items():
                if line.startswith(label):
                    found[key] = int(float(line.split(":")[-1].strip().rstrip(".")))
        if len(found) != 3:
            return False
        scores = [found["instruct"], found["vis"], found["cons"]]
        if any(score not in range(1, 6) for score in scores):
            return False
        return scores
    except (ValueError, TypeError):
        return False


def parse_json(out):
    """解析 JSON 格式，容忍 ```json 围栏和前后多余文本。"""
    cleaned = (out or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    obj = None
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                return False
    if not isinstance(obj, dict):
        return False
    try:
        scores = [int(obj[key]) for key in SCORE_KEYS]
    except (KeyError, TypeError, ValueError):
        return False
    if any(score not in range(1, 6) for score in scores):
        return False
    return scores


JSON_INSTRUCTION = (
    "\n\nReturn JSON only with keys: reason, instruction_compliance, "
    "visual_quality_stability, consistency_detail_fidelity."
)
