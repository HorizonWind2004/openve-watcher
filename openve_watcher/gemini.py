"""原生 Gemini 视频打分：两个完整视频直接上传，不抽帧。

沿用 KiwiEdit 官方脚本的两文件输入协议（先原始视频、后编辑视频），
与 I2V-transfer 的 ``evaluation/gemini_video.py`` 行为一致。
不提供抽帧代理回退——两种协议的分数不可混用，静默回退会污染结果。
"""

import logging
import time

LOGGER = logging.getLogger(__name__)
MAX_ATTEMPTS = 3


def create_client(api_key, timeout):
    from google import genai

    return genai.Client(api_key=api_key, http_options={"timeout": int(timeout * 1000)})


def evaluate_video_pair(original, edited, prompt, *, api_key, model, timeout, parse_response):
    """上传两个视频、等待处理完成、打分，然后无论成败都删除远端文件。

    返回 ``(scores, raw_text)``。三次尝试都拿不到合法三项分数则抛异常，
    由调用方决定是跳过还是稍后重试——这里不吞异常。
    """
    uploaded = []
    with create_client(api_key, timeout) as client:
        try:
            for path in (original, edited):
                uploaded.append(client.files.upload(file=str(path)))

            deadline = time.monotonic() + timeout
            for index, video in enumerate(uploaded):
                while getattr(video.state, "name", video.state) == "PROCESSING":
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Gemini 视频处理超时: {video.name}")
                    time.sleep(min(2, max(0, deadline - time.monotonic())))
                    video = client.files.get(name=video.name)
                if getattr(video.state, "name", video.state) != "ACTIVE":
                    raise RuntimeError(f"Gemini 视频处理失败: {video.name} ({video.state})")
                uploaded[index] = video

            last_error = None
            for attempt in range(MAX_ATTEMPTS):
                try:
                    response = client.models.generate_content(
                        model=model,
                        config={"system_instruction": prompt},
                        contents=uploaded,
                    )
                    text = response.text or ""
                    scores = parse_response(text)
                    if scores:
                        return scores, text
                    raise ValueError("Gemini 返回的三项评分不合法")
                except Exception as exc:  # noqa: BLE001 - 重试所有可恢复错误
                    last_error = exc
                    if attempt < MAX_ATTEMPTS - 1:
                        time.sleep(2)
            raise RuntimeError(f"Gemini 打分连续 {MAX_ATTEMPTS} 次失败") from last_error
        finally:
            # 第二个上传失败时，第一个也要删掉，否则会在配额里留垃圾文件。
            for video in uploaded:
                try:
                    client.files.delete(name=video.name)
                except Exception:  # noqa: BLE001 - 清理失败不该盖掉真正的错误
                    LOGGER.warning("未能删除 Gemini 远端文件 %s", getattr(video, "name", "?"))
