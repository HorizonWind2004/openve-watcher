"""通过 HuggingFace 把 OpenVE 推理与 Gemini 打分拆到两台机器上。

集群侧只产出视频并上传；打分侧只消费 HuggingFace，不需要 OpenVE-Bench、
不需要 bench CSV、也不需要访问集群。两侧唯一的契约是每个样本目录里的
``meta.json``（见 :mod:`openve_watcher.manifest`）。
"""

__all__ = ["manifest", "parse", "prompts"]
