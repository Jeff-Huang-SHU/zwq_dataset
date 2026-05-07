# Copyright (c) 本仓库复现扩展；数据与 Fast-DetectGPT 算法版权归原项目与数据提供方。
"""
只读加载 ``new_data.json``，按 ``tittle`` 将人类与机器文本展开为
``{"original": [...], "sampled": [...]}``，结构与 ``data_builder`` 写入的
``.raw_data.json`` 中字段一致（见 ``PAIRING_SPEC.md``）。

**不得**改写源 JSON 文件；本模块仅 ``open(..., "r")`` 与内存构造。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, TypedDict


class PairingStats(TypedDict):
    n_records: int
    n_titles: int
    n_pairs: int
    n_skipped_only_human_titles: int
    n_skipped_only_ai_titles: int
    n_skipped_multi_human_titles: int
    n_rows_skipped_human: int
    n_rows_skipped_ai: int


def load_records(path: str) -> list[dict[str, Any]]:
    """从磁盘读取 ``new_data.json`` 顶层数组（只读）。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fast_detect_pair_lists(
    records: list[dict[str, Any]],
    *,
    human_model_value: str = "h",
) -> tuple[dict[str, list[str]], PairingStats, dict[str, list[str]]]:
    """
    构造与 ``scripts/fast_detect_gpt.py`` / ``data_builder.load_data`` 兼容的
    ``original``（人类）与 ``sampled``（机器）等长列表。

    配对规则（摘要，完整见 ``PAIRING_SPEC.md``）：
    - 按 ``tittle`` 分组；
    - 若该标题下恰有 1 条人类（``model == human_model_value``）与至少 1 条机器，
      则对该标题下每条机器文本生成一对 ``(人类正文, 机器正文)``；
    - 标题顺序：按该标题在文件中**首次出现**的行号升序；
    - 同一标题下机器文本顺序：按行号升序。

    返回：
    - ``data``：``{"original": [...], "sampled": [...]}``
    - ``stats``：计数信息
    - ``skipped``：被跳过的标题列表（键：``only_human`` / ``only_ai`` / ``multi_human``）
    """
    by_title: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"idx_h": [], "idx_ai": []}
    )
    for i, row in enumerate(records):
        title = row["tittle"]
        if row.get("model") == human_model_value:
            by_title[title]["idx_h"].append(i)
        else:
            by_title[title]["idx_ai"].append(i)

    title_min_index: dict[str, int] = {}
    for title, g in by_title.items():
        all_i = g["idx_h"] + g["idx_ai"]
        title_min_index[title] = min(all_i) if all_i else 0

    sorted_titles = sorted(by_title.keys(), key=lambda t: title_min_index[t])

    original: list[str] = []
    sampled: list[str] = []
    skipped: dict[str, list[str]] = {
        "only_human": [],
        "only_ai": [],
        "multi_human": [],
    }
    for title in sorted_titles:
        g = by_title[title]
        idx_h = sorted(g["idx_h"])
        idx_ai = sorted(g["idx_ai"])
        if not idx_h and idx_ai:
            skipped["only_ai"].append(title)
            continue
        if idx_h and not idx_ai:
            skipped["only_human"].append(title)
            continue
        if not idx_h and not idx_ai:
            continue
        if len(idx_h) > 1:
            skipped["multi_human"].append(title)
        # 多人类时与规格书一致：仅使用文件中最前一条人类，其余人类行既不配对也不重复输出警告外数据
        h_idx = idx_h[0]
        human_text = records[h_idx]["content"]
        for j in idx_ai:
            original.append(human_text)
            sampled.append(records[j]["content"])

    stats: PairingStats = {
        "n_records": len(records),
        "n_titles": len(by_title),
        "n_pairs": len(original),
        "n_skipped_only_human_titles": len(skipped["only_human"]),
        "n_skipped_only_ai_titles": len(skipped["only_ai"]),
        "n_skipped_multi_human_titles": len(skipped["multi_human"]),
        "n_rows_skipped_human": sum(len(by_title[t]["idx_h"]) for t in skipped["only_human"]),
        "n_rows_skipped_ai": sum(len(by_title[t]["idx_ai"]) for t in skipped["only_ai"]),
    }
    return {"original": original, "sampled": sampled}, stats, skipped


def assert_fast_detect_shape(data: dict[str, list[str]]) -> None:
    o, s = data["original"], data["sampled"]
    assert len(o) == len(s) and len(o) > 0, "original/sampled 须非空且等长"


def main() -> None:
    parser = argparse.ArgumentParser(description="从 new_data.json 构造 Fast-DetectGPT 配对数据")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/hjf/A_workspace/detect-gpt/new_data.json",
        help="只读数据源路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="若非空，将写入 ``<path>.raw_data.json``（与 data_builder 相同命名约定）",
    )
    parser.add_argument("--human_tag", type=str, default="h", help="人类样本的 model 字段取值")
    args = parser.parse_args()

    records = load_records(args.input)
    data, stats, skipped = build_fast_detect_pair_lists(records, human_model_value=args.human_tag)
    assert_fast_detect_shape(data)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if any(skipped[k] for k in skipped):
        print("skipped_titles:", json.dumps(skipped, indent=2, ensure_ascii=False))

    if args.output:
        out = args.output
        if not out.endswith(".raw_data.json"):
            path = f"{out}.raw_data.json"
        else:
            path = out
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("written:", path)


if __name__ == "__main__":
    main()
