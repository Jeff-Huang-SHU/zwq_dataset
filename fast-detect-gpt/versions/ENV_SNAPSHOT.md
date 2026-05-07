# 环境快照（T0）

**记录时间**：2026-05-05（以本机会话日期为准）  
**项目路径**：`/home/hjf/A_workspace/fast-detect-gpt`  
**关联数据（只读）**：`/home/hjf/A_workspace/detect-gpt/new_data.json`

---

## 1. 代码版本

| 项 | 值 |
|----|-----|
| Git 仓库 | **当前目录不是 git 仓库**（`git rev-parse HEAD` 不可用）；若上游为克隆仓库，请在本机执行 `git rev-parse HEAD` 并补登于此 |
| 建议 | 将本目录 `git init` 并关联远端，或从官方仓库重新 clone 后重新采集本快照 |

---

## 2. 操作系统与硬件

| 项 | 值 |
|----|-----|
| 内核 / OS | `Linux ubuntu 6.14.0-36-generic #36~24.04.1-Ubuntu SMP … x86_64` |
| GPU | `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` |
| 显存 | `97887 MiB` |
| 驱动 | `580.82.07` |

---

## 3. Python 解释器

| 项 | 值 |
|----|-----|
| 解释器 | `/home/hjf/A_workspace/fast-detect-gpt/.venv/bin/python` |
| 版本 | **Python 3.12.3** |

---

## 4. 关键 Python 依赖（`.venv` 实测）

以下由 `pip show` 在 **2026-05-05** 于上述 venv 中读取：

| 包 | 实测版本 |
|----|-----------|
| torch | **2.11.0** |
| transformers | **5.7.0** |
| datasets | **4.8.5** |
| numpy | **2.4.4** |

### PyTorch / CUDA（运行时探测）

| 项 | 值 |
|----|-----|
| `torch.version.cuda` | **13.0** |
| cuDNN | **91900** |
| `torch.cuda.is_available()` | **True** |

---

## 5. 与仓库 `requirements.txt` 的差异（重要）

仓库内 `requirements.txt` 写明：

- `transformers==4.28.1`
- `datasets==2.12.0`

当前 **`.venv` 中版本更高**（transformers 5.7.0、datasets 4.8.5）。这会导致：

- 与论文 README「Python3.8 + PyTorch1.10」及老版 API 行为不完全一致；
- 后续若需「严格对齐论文复现」，建议在**独立 conda/venv** 中按 `requirements.txt` 重装并另写一份 `ENV_SNAPSHOT.md`。

**复现策略建议**：

1. **主线实验（当前机）**：继续用本快照环境，所有结果文件注明本 `ENV_SNAPSHOT.md` 路径与日期。  
2. **严格论文环境**：新建环境，`pip install -r requirements.txt` 后再采一份快照并对比 AUC。

---

## 6. 可复现命令（供他人校验）

```bash
cd /home/hjf/A_workspace/fast-detect-gpt
.venv/bin/python -V
.venv/bin/pip show torch transformers datasets numpy
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

---

## 7. 下一步（T1）

在确认本快照可接受的前提下，进行 `new_data.json` 只读加载与配对构造（见 `versions/复现计划与校验清单.md` 中 T1）。
