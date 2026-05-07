# `new_data.json` → Fast-DetectGPT 配对规格（T2）

**数据源（只读）**：`/home/hjf/A_workspace/detect-gpt/new_data.json`  
**实现**：[`load_new_data_pairs.py`](load_new_data_pairs.py)  
**约束**：不修改源 JSON 的字段名、结构与内容；仅按本规格在内存（或可选导出文件）中构造 `original` / `sampled`。

---

## 1. 源数据字段（约定）

每条记录为 JSON 对象，至少使用：

| 字段 | 用途 |
|------|------|
| `tittle` | 标题键，用于将「同一题目的人类稿」与「多模型机器稿」归为一组（字段拼写以源文件为准）。 |
| `content` | 正文，作为 `original` / `sampled` 的字符串来源。 |
| `model` | 人类：`"h"`；机器：任意非 `"h"` 的取值。 |

---

## 2. 分组键

- 使用 **`tittle` 字符串完全相等** 作为分组键（**不做**去首尾空格、全半角或 Unicode 规范化）。
- 因此，若同一语义题目在源文件中出现两种写法（例如带前导空格与不带），会被视为**两个不同标题**；其中一组可能仅有机器、另一组仅有「人类」，从而**无法配对**。当前数据集中存在此类情况（见 §5 实测列表）。

---

## 3. 配对规则

1. 将全部记录按 `tittle` 分组。
2. 对每个 `tittle`：
   - 若 **人类行数 = 0** 且 **机器行数 ≥ 1**：该标题下所有机器行**不参与**任何配对（记为「仅机器」标题）。
   - 若 **人类行数 ≥ 1** 且 **机器行数 = 0**：该标题下人类行**不参与**配对（记为「仅人类」标题）。
   - 若 **人类行数 ≥ 1** 且 **机器行数 ≥ 1**：
     - **人类正文**：取该组内**文件行号最小**的人类记录的 `content`（当前数据集中每组人类恒为 1 条；若将来出现多条，本实现仅取最前一条，与代码注释一致）。
     - **机器正文**：对该组内每条**机器**记录各生成一对：  
       `original.append(人类正文)`，`sampled.append(该机器 content)`。
3. **标题遍历顺序**：按各标题在源文件中**首次出现**的行号升序（与 `load_new_data_pairs.py` 中 `title_min_index` 一致）。
4. **同一标题内机器顺序**：按该行在源文件中的行号升序。

输出结构：

```json
{
  "original": ["人类正文", ...],
  "sampled": ["机器正文", ...]
}
```

要求：`len(original) == len(sampled) > 0`，且第 `i` 对表示「同一 `tittle` 下」人类与第 `i` 个（按 §3.4 排序）机器样本。

---

## 4. 与原版 `data_builder` 的对应关系

- `data_builder.save_data` 写入的 `.raw_data.json` 使用相同顶层键：`original`、`sampled`（见 `scripts/data_builder.py`）。
- `scripts/fast_detect_gpt.py` 通过 `load_data` 读取该结构；本配对结果可直接存为 `*.raw_data.json` 后由原版脚本读取（路径前缀不含后缀）。

---

## 5. 当前数据集实测（2026-05-05，以脚本输出为准）

| 统计项 | 值 |
|--------|-----|
| 总记录数 | 14000 |
| 不同 `tittle` 数 | 505 |
| 生成配对数 `n_pairs` | **13425** |
| 含人类且含机器的标题数 | 499 |
| 仅人类标题数 | 1 |
| 仅机器标题数 | 5 |
| 未配对的机器行数合计 | 75 |
| 未配对的人类行数合计 | 1 |

**仅人类标题（1）**：

- `" 4e− 电子化合物的制备及其电输运特性"`（注意前导空格）

**仅机器标题（5）**（节选说明）：

- 其中一条为 `"4e− 电子化合物的制备及其电输运特性"`（与上项语义相近但 **与带空格版本非同一键**），导致人类与机器被拆到两组。

若业务上希望合并此类标题，属于**数据清洗/规范化**，须单独导出中间表且**不得**在原 `new_data.json` 上就地改写；本复现流程默认严格按 §2 原样键匹配。

---

## 6. 正确性自检命令

```bash
cd /home/hjf/A_workspace/fast-detect-gpt
.venv/bin/python versions/load_new_data_pairs.py \
  --input /home/hjf/A_workspace/detect-gpt/new_data.json
```

期望：`n_pairs == 13425`，`n_records == 14000`，`n_rows_skipped_human == 1`，`n_rows_skipped_ai == 75`。

可选写出与原版兼容的文件（**新路径**，不覆盖源数据）：

```bash
.venv/bin/python versions/load_new_data_pairs.py \
  --input /home/hjf/A_workspace/detect-gpt/new_data.json \
  --output /home/hjf/A_workspace/fast-detect-gpt/versions/cache/new_data_paired
# 生成 versions/cache/new_data_paired.raw_data.json
```

---

## 7. 修订历史

| 日期 | 说明 |
|------|------|
| 2026-05-05 | 初版：与 `load_new_data_pairs.py` 首版行为对齐。 |
