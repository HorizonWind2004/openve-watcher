"""逐字节断言评分提示词与官方 Kiwi-Edit 一致。

基准是 tests/data/kiwi_eval_prompts.py，那是官方 eval_openve_gemini.py 的
逐字副本。这些测试失败时，正确做法是同步 prompts.py 去贴合官方，
而不是改基准来让测试变绿——评分标准一改，分数就不可比了。
"""

import pytest

from openve_watcher import prompts
from openve_watcher.watch import system_prompt
from tests.data import kiwi_eval_prompts as official

OFFICIAL_CONSTANTS = sorted(
    name for name in dir(official) if name.isupper() and isinstance(getattr(official, name), str)
)
# OpenVE-Bench 七类计划实际用到的类别。
PLANNED_TYPES = [
    "global_style",
    "local_change",
    "background_change",
    "local_remove",
    "local_add",
    "creative_edit",
    "subtitle_edit",
]


def test_official_fixture_is_the_expected_shape():
    assert len(OFFICIAL_CONSTANTS) == 8
    assert sorted(official.prompt_type) == [
        "background_change",
        "global_style",
        "local_add",
        "local_change",
        "local_remove",
    ]


@pytest.mark.parametrize("name", OFFICIAL_CONSTANTS)
def test_every_official_constant_is_byte_identical(name):
    assert hasattr(prompts, name), f"缺少官方常量 {name}"
    assert getattr(prompts, name) == getattr(official, name), f"{name} 与官方不逐字一致"


@pytest.mark.parametrize("key", sorted(official.prompt_type))
def test_official_keys_map_to_the_same_prompt(key):
    assert prompts.prompt_type[key] == official.prompt_type[key]


def test_prompt_type_only_contains_official_prompts():
    """prompt_type 里每条提示词都必须能在官方常量里找到原文。"""
    official_texts = {getattr(official, name) for name in OFFICIAL_CONSTANTS}
    for key, text in prompts.prompt_type.items():
        assert text in official_texts, f"{key} 用的提示词不在官方常量里"


def test_non_official_prompts_are_kept_out_of_prompt_type():
    official_texts = {getattr(official, name) for name in OFFICIAL_CONSTANTS}
    assert prompts.NON_OFFICIAL_PROMPT_TYPE, "非官方类别应当被保留但隔离"
    for key, text in prompts.NON_OFFICIAL_PROMPT_TYPE.items():
        assert key not in prompts.prompt_type, f"{key} 不该出现在 prompt_type 里"
        assert text not in official_texts, f"{key} 其实是官方提示词，应移进 prompt_type"


@pytest.mark.parametrize("edited_type", PLANNED_TYPES)
def test_planned_types_are_all_covered(edited_type):
    assert edited_type in prompts.prompt_type


@pytest.mark.parametrize("edited_type", PLANNED_TYPES)
def test_system_prompt_matches_official_construction(edited_type):
    """system prompt 的构造必须等于官方那一行 format(edit_prompt=prompt)。"""
    edit_prompt = "把天空换成紫色的黄昏"
    expected = prompts.prompt_type[edited_type].format(edit_prompt=edit_prompt)
    assert system_prompt(edited_type, edit_prompt, json_mode=False) == expected


def test_json_mode_is_the_only_deviation_and_is_additive():
    """--json-mode 是我们额外加的；它只在官方文本后面追加，不改动原文。"""
    plain = system_prompt("global_style", "x", json_mode=False)
    with_json = system_prompt("global_style", "x", json_mode=True)
    assert with_json.startswith(plain)
    assert len(with_json) > len(plain)
