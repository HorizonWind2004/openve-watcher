from openve_watcher import parse


def test_parse_plain_returns_fixed_order_not_response_order():
    """返回顺序恒为 [指令遵从, 画质稳定性, 一致性]，与响应里的行序无关。"""
    text = (
        "Instruction Compliance: 4\n"
        "Consistency & Detail Fidelity: 3\n"
        "Visual Quality & Stability: 5.\n"
    )
    assert parse.parse_plain(text) == [4, 5, 3]


def test_parse_plain_and_json_agree_on_order():
    plain = parse.parse_plain(
        "Instruction Compliance: 1\nVisual Quality & Stability: 2\nConsistency & Detail Fidelity: 3\n"
    )
    js = parse.parse_json(
        '{"instruction_compliance": 1, "visual_quality_stability": 2, "consistency_detail_fidelity": 3}'
    )
    assert plain == js == [1, 2, 3]


def test_parse_plain_rejects_incomplete():
    assert parse.parse_plain("Instruction Compliance: 4\n") is False


def test_parse_plain_rejects_out_of_range():
    text = "Instruction Compliance: 9\nVisual Quality & Stability: 3\nConsistency & Detail Fidelity: 3\n"
    assert parse.parse_plain(text) is False


def test_parse_json_handles_fence_and_prose():
    text = """好的，结果如下：
```json
{"reason": "ok", "instruction_compliance": 5,
 "visual_quality_stability": 4, "consistency_detail_fidelity": 3}
```"""
    assert parse.parse_json(text) == [5, 4, 3]


def test_parse_json_rejects_missing_key():
    assert parse.parse_json('{"instruction_compliance": 5}') is False


def test_parse_json_rejects_non_object():
    assert parse.parse_json("[1, 2, 3]") is False


def test_parse_json_rejects_out_of_range():
    text = '{"instruction_compliance": 0, "visual_quality_stability": 3, "consistency_detail_fidelity": 3}'
    assert parse.parse_json(text) is False


def test_parse_handles_none():
    assert parse.parse_plain(None) is False
    assert parse.parse_json(None) is False
