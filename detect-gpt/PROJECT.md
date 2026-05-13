# detect-gpt 项目说明

本仓库在经典 **DetectGPT**（扰动 + 对数似然对比）流程上做了扩展，面向 **中文语料**、**paired JSON 人机对** 以及 **按 AI 模型（subkey）筛选** 的实验与评估。

---

## 1. 核心目标

- **区分**：人类撰写文本（`original`）与机器生成文本（`sampled`）。
- **主方法**：对文本做 **span 遮罩 + T5/mT5 填空扰动**，在 **因果语言模型** 上比较扰动前后的 **log-likelihood**，得到 `d` / `z` 等分数，再算 ROC/PR。
- **基线**：似然、rank、log-rank、entropy，以及两条 **中文 RoBERTa 序列分类** 监督检测器。

---

## 2. 目录与主要文件

| 路径 | 说明 |
|------|------|
| `detect_gpt.py` | **主入口**（当前维护版本）：中文 jieba/char 扰动、`paired_json`、`--subkey`、结果目录与 summary 等。 |
| `custom_datasets.py` | 内置 HF 外数据集名（writing / english / german / pubmed 等）的加载。 |
| `process_result.py` | 从 `*_results.json` 的 `predictions` 计算 accuracy / F1 / ROC AUC / precision / recall（及 PR AUC），可选 F1 最优阈值。 |
| `run.sh` | 对 `new_data.json` 中多种 AI `model` 各跑一遍 `detect_gpt.py`（不同 `--subkey`）。 |
| `temp.py` | 临时脚本：校验某次 `raw_data.json` 中 `sampled` 是否均能在 `new_data.json` 中对应到指定 `subkey` 的 model。 |
| `new_data.json` | 配对语料示例：列表对象，含 `register`、`tittle`、`content`、`model` 等。 |
| `requirements.txt` | `torch`、`transformers`、`datasets`、`matplotlib`、`tqdm`、`scikit-learn`、`openai`。主脚本还依赖 **`jieba`**（需自行 `pip install jieba`）。 |
| `versions/` | 历史/变体脚本（如 `1_data_run.py`、`3_temp_run.py`）；日常以根目录 **`detect_gpt.py`** 为准。 |
| `tmp_results/` → `results/` | 单次运行先写入 `tmp_results/...`，结束后 **`os.rename` 到 `results/...`**。 |
| `原文的信息/` | 与论文/原文相关的辅助脚本，不参与主流程。 |

---

## 3. 数据与 `paired_json`

### 3.1 `--dataset paired_json`

- 需同时指定 **`--paired_json_path`**（相对仓库根的路径，如 `new_data.json`）。
- JSON 为 **对象数组**。按 **`(register, tittle)`** 分组（注意字段名为 **`tittle`**）。
- **`model` 转小写为 `h`**：视为 **人类**，进入 `original` 候选。
- **其余 `model`**：视为 **AI**，进入 `sampled` 候选。

### 3.2 `--subkey`（可选，默认空）

- 若 **非空**：同一组内 **仅** `str(model).strip()` 与 `subkey`（strip 后）**完全一致** 的条目可作为 AI；每个 human 仍在该组的 AI 候选中 **随机** 抽一条配对。
- 若某组 **有 human 但没有匹配 subkey 的 AI**：整组跳过。
- 若 **无 human 仅有 AI**：不形成 pair（跳过）。

配对后会对齐长度（`trim_to_shorter_length`，基于 tokenizer 子词长度），再按 **`--max_pair_tokens`** 过滤超长对，最后若总对数大于 **`--n_samples`** 再随机下采样。

### 3.3 其它 `--dataset`

与 `custom_datasets` 或 HuggingFace `datasets` 加载后，可走 **模型续写生成 `sampled`** 的经典路径（非 `paired_json` 时）。

---

## 4. 运行流程概要（`detect_gpt.py`）

1. 解析参数；**默认 HF 缓存根**：`<detect-gpt 上一级>/cache`**，可用 **`--cache_dir`** 覆盖。
2. 加载 **base 因果 LM**、**mask 填空模型**（默认 Qwen2.5 + `google/mt5-base`）、GPT2 tokenizer（部分逻辑/OpenAI token 估算用）。
3. **`generate_data`**：`paired_json` → `load_paired_json_data`；否则语料 shuffle、过滤、**`generate_samples`** 续写得到 `original` / `sampled`。
4. 可选 **`--scoring_model_name`**：与 base 分离的打分模型。
5. 写出 **`raw_data.json`**、`args.json`。
6. **基线**（除非 `--skip_baselines`）：likelihood、rank、log-rank、entropy、两条 supervised 中文 RoBERTa。
7. **扰动实验**（除非 `--baselines_only`）：对 **`--n_perturbation_list`** 中每个 n，跑 `d` 与 `z`，写出 **`perturbation_{n}_d_results.json`** / **`perturbation_{n}_z_results.json`**。
8. **ROC/PR 曲线图**（`roc_curves.png`、likelihood ratio 相关直方图等）。
9. 将 **`tmp_results/...` 整目录改名为 `results/...`**。
10. **`write_perturbation_summaries`**：对每个 `perturbation_*_[dz]_results.json` 调用 **`process_result.compute_result_metrics`**，将 **`perturbation_*_[dz]_summary.json`** 写入同次结果目录下的 **`metrics/`** 子目录。

结果 JSON 中会通过 **`attach_subkey`** 附带本次 **`subkey`** 字段（与 `args.json` 一致）。

---

## 5. 输出目录命名

结果路径大致为：

`tmp_results/{output_name/}{base_model}-{scoring?}-{mask_model}-{sampling}/`  
`{日期}-{时间}-{fp32|fp16|int8}-{pct_masked}-{n_perturb_rounds}-{dataset}-{n_samples}/`

- **`--output_name`**：便于区分多次实验（如不同 `subkey` 批跑时放在子目录名里）。
- 结束后同名树出现在 **`results/`** 下。

---

## 6. `process_result.py` 与 `metrics/*_summary.json`

- 命令行：`--input` 指向某次 **`perturbation_*_results.json`**，`--output` 写汇总 JSON。
- 指标含 **ROC AUC、PR AUC（average_precision）、F1 最优阈值下的 accuracy / precision / recall / F1** 等；若原始分数方向导致 ROC AUC &lt; 0.5，可对分数自动取反（见脚本内说明）。

---

## 7. 批量实验：`run.sh`

对 `new_data.json` 中出现的多种 AI **`model`** 各执行一次 `detect_gpt.py`（通过 **`--subkey`**）。后台长时间运行示例：

```bash
nohup bash run.sh > run.log 2>&1 &
```

**注意**：多行 shell 续行时，**除最后一行外，行尾必须有 `\`**，否则下一行不会拼进同一条命令（例如 `--subkey` 会丢失）。

---

## 8. 环境与常见问题

- **GPU**：脚本中设备为 **CUDA**；无卡需自行改代码或环境。
- **`n_samples` 与 `batch_size`**：基线循环步数为 `n_samples // batch_size`，若商为 0 会导致 ROC 计算异常；至少保证 **`n_samples >= batch_size`** 或减小 `batch_size`。
- **Hugging Face 下载**：若配置 **`HF_ENDPOINT`**（如镜像）超时，可能影响首次拉取 `gpt2`/大模型；可换端点、代理或预先把模型缓存在 **`--cache_dir`** 指向的目录。

---

## 9. 与上游 DetectGPT 的关系

- 思想一致：**扰动文本 → 比较打分模型上的似然变化**。
- 本仓库增强：**中文分词/字级 span**、**paired JSON 与 subkey**、**中文监督基线**、**默认缓存与结果目录习惯**、**跑完自动生成 perturbation 的 summary** 等。

若需最小复现论文原版行为，可参考 `versions/` 内较早脚本或上游公开实现，并对齐超参与数据管线。
