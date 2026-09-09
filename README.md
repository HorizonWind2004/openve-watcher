# openve-watcher

爬 HuggingFace 上的 OpenVE 推理结果，对新出现的样本自动跑 Gemini 打分。

**这是客户端。** 服务端（跑推理、往 HF 上传的那一侧）由我们运行——我们往 HF 传一个
public dataset 仓库，这个爬虫就能自动识别到并开始打分。你只需要一个 HF token
和一个 Gemini key，**不需要 GPU、不需要 OpenVE-Bench、不需要访问我们的集群**。

为什么要拆：跑推理的 GPU 集群没有 Gemini 的网络访问和 API key，而有 key 的机器
没有 GPU。两边各做自己能做的事，互相不需要 SSH、不需要共享文件系统。

```
集群（有 GPU，没 Gemini key）          HuggingFace              打分机（有 Gemini key，没 GPU）
────────────────────────────          ───────────              ──────────────────────────────
i2v-opd 推理 → 视频落盘                                          
        ↓                                                        
   openve-push  ──────────────→  sanaka87/openve_xxx  ←────────  openve-watch
   （发现完整样本就上传）            (public dataset)              （发现新样本就打分）
                                          ↑                              ↓
                                   scores/*.jsonl  ←──────────  --push-scores 回传
```

## HuggingFace 仓库布局

每个推理 run 一个 dataset 仓库，**每个样本一个自包含目录**：

```
run.json                                      本次推理的运行级信息
samples/<edited_type>/<base>/meta.json         类别 + 编辑指令
samples/<edited_type>/<base>/original.mp4      原始视频
samples/<edited_type>/<base>/edited.mp4        模型编辑后的视频
scores/<model>_gemini_video_score.jsonl        打分侧回传（可选）
```

打分侧因此**不需要 OpenVE-Bench、不需要 benchmark CSV、不需要访问集群**，
只要一个 HF token 和一个 Gemini key。

刻意不用集中式 manifest 文件：那需要读-改-写，两个推送进程或一次中断就能写坏它。
用「一个样本一个 meta.json」，仓库的文件列表本身就是清单，新增永远是纯追加。

## 安装

两侧装的东西不一样。

```bash
# 客户端（打分侧，你跑的）：需要 Gemini SDK
pip install -e '.[score]'

# 服务端（集群侧，我们跑的）：只要 huggingface_hub，不装 Gemini SDK
pip install -e .
```

> **集群上千万别往共享虚拟环境里 pip 装东西。** 我们踩过一次：装 flash-attn 时 pip
> 顺手把 cuDNN 从 9.25.1.1 降到 9.10.2.21，教师侧 Conv3d 静默回退到
> `slow_conv_dilated3d`，训练慢了两倍多，排查了几个小时。装到独立 target 或独立 venv。

## 客户端：爬取并打分（你要跑的就是这个）

```bash
export HF_TOKEN=hf_xxx
export GEMINI_API_KEY=key1,key2      # 或用 --gemini-key-file，每行一个

pip install -e '.[score]'

openve-watch \
  --author sanaka87 \
  --prefix openve_ \
  --out-dir ./openve-scores \
  --max-workers 4 \
  --push-scores \
  --interval 300
```

挂着不用管。我们每传一个新 run，它下一轮就会发现并开始打分；同一个 run 里
推理边跑边传，它也会持续把新样本补上。

- `--push-scores` 把分数回传到同一个 HF 仓库，这样我们那边不用 key 也能读到结果。
- `--once` 只跑一轮；`--limit N` 每个仓库每轮最多打 N 条（试水用）。
- 单条失败只打到 stderr，不中断整轮，下一轮会自动重试。

### 本地进度台账

每个仓库在 `--out-dir` 下有两个文件：

```
openve-scores/sanaka87__openve_33f_fa_cfg1/
├── progress.txt                                 一行一条，只记「做过没做过」
└── gemini-2.5-pro_gemini_video_score.jsonl      完整结果，含 Gemini 原始回复
```

`progress.txt` 长这样，简单到可以直接 `wc -l` 数进度：

```
sanaka87/openve_33f_fa_cfg1	global_style/0000_global_style_Apply_the_Impression
sanaka87/openve_33f_fa_cfg1	local_add/0012_local_add_Add_a_cat
```

两份台账**取并集**判定已完成，一个防重复、一个防漏做：

- **防重复打分**：任一文件说做过就跳过。jsonl 删了也不会重打，因为 txt 还在。
- **防漏打分**：写入顺序固定「先 jsonl，后 txt」。若在两次写入之间崩溃，
  txt 缺这条但 jsonl 有，下一轮 `reconcile` 会把它补回 txt——**反过来永远不会发生**，
  所以不存在「txt 说做过、其实没打」的情况。
- **失败的样本不进台账**，下一轮必然重试。
- 每轮都打印 `仓库 N 条 / 已打分 M 条 / 待打分 K 条`，漏没漏一眼就看得出来。
- txt 里坏掉的行会被忽略而不是让程序崩——这个文件本来就是崩溃现场的产物。

要重打某一条：从 `progress.txt` 里删掉那一行，并从 jsonl 里删掉对应的行。

---

## 服务端：上传（我们这侧运行，你不用管）

```bash
export HF_TOKEN=hf_xxx

openve-push \
  --inference-dir /export/home4/yanhao/runs/inference/openvebench/openve_33f_fa_cfg1 \
  --repo sanaka87/openve_33f_fa_cfg1 \
  --bench-csv "$I2V_CACHE_ROOT/OpenVE-Bench/benchmark_videos.csv" \
  --bench-dir "$I2V_CACHE_ROOT/OpenVE-Bench" \
  --watch --interval 60
```

- `--watch` 让它常驻，推理边跑边传；不加就只跑一轮。
- 先用 `--dry-run` 看要传什么，什么都不会上传。
- 默认建 **public** 仓库；要私有加 `--private`。
- `--collection <slug>` 可以顺手把仓库加进 collection（失败不阻塞上传）。

**什么算「完整样本」**：主视频和 `*_info.txt` 同时存在。`_info.txt` 是推理侧最后
写的文件，只有它落盘才说明这条真的做完了。`_compare.mp4`（三联对比）和
`_reason_edit.mp4`（多一帧推理帧）是派生产物，**不会**被当成待打分的主视频。

**幂等**：每轮都先读仓库文件列表，只传缺的。重复运行不会重传。
每个样本一次 `upload_folder`（三个文件一个 commit），所以打分侧看不到半个样本。

## 与官方 Kiwi-Edit 的对齐

评分提示词严格对齐 [showlab/Kiwi-Edit](https://github.com/showlab/Kiwi-Edit)
的 `eval_openve_gemini.py`：

| 项目 | 状态 |
|---|---|
| 8 个官方 prompt 常量 | **逐字节一致**，由 `tests/test_prompts_alignment.py` 断言 |
| system prompt 构造 | `prompt_type[edited_type].format(edit_prompt=prompt)`，与官方同一行 |
| Gemini 调用 | `config={"system_instruction": ...}`，`contents=[原始视频, 编辑视频]`（顺序同官方）|
| 输入协议 | 两个**完整视频**直接上传，不抽帧、无代理回退 |
| 重试 | 3 次，与官方一致 |

基准是 `tests/data/kiwi_eval_prompts.py`——官方文件的逐字副本。
**这些测试失败时，要改的是 `prompts.py`，不是基准。** 评分标准一动，分数就不可比了。

### 已知的两处偏离，都是刻意的

1. **`prompt_type` 多了 5 个键。** 官方只映射 5 类
   (`global_style` / `local_change` / `background_change` / `local_remove` / `local_add`)，
   遇到 `creative_edit`、`subtitle_edit` 会直接 KeyError，但这两类的**常量在官方文件里是有的**。
   OpenVE-Bench 的七类计划需要它们，所以补进了映射，用的仍是官方原文。
   `camera_edit` / `camera_multi_shot_edit` 同理。

2. **官方没有提示词的 7 个类别被隔离了。** `color_change`、`texture_change`、
   `shape_change`、`relight`、`action_edit`、`move_edit`、`scale_edit` 的提示词不是官方的，
   放在 `prompts.NON_OFFICIAL_PROMPT_TYPE` 里，**不在** `prompt_type` 中。
   打分默认走 `prompt_type`，所以不可能静默用上非官方标准。要用得显式合并。

3. **`--json-mode` 是我们加的**，默认关闭。它只在官方文本**后面追加**一句要求返回 JSON，
   不改动官方原文（有测试断言这一点）。要完全对齐官方就别加这个开关。

## 测试

```bash
pip install -e '.[dev]'
pytest -q          # 80 passed，全部离线，不碰 HF 也不碰 Gemini
```

HF 和 Gemini 都用假对象替掉，所以测试可以在 CI 里跑。覆盖的关键行为：
半个样本被跳过、重复上传幂等、只剩 txt 也不会重复调用 Gemini、
只剩 jsonl 会补回 txt 并跳过、失败的样本不进台账因而会重试、
`meta.json` 与目录类别不一致时直接失败、派生视频不被误当主视频、
以及上面那张对齐表的每一行。
