#!/usr/bin/env python3
"""临时校验：raw_data.json 里 sampled 是否均来自 new_data 中 args.subkey 对应的 model。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def norm(t: str) -> str:
    """与 detect_gpt.load_paired_json_data 中 strip_newlines 一致。"""
    return " ".join(str(t).split())


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument(
        "--args-json",
        type=Path,
        default=root
        / "tmp_results/deepseek-v3.2/Qwen_Qwen2.5-1.5B-Instruct-google/mt5-base-temp/2026-05-11-15-41-10-050459-fp32-0.15-1-paired_json-512/args.json",
        help="含 subkey 的 args.json",
    )
    p.add_argument(
        "--raw-data",
        type=Path,
        default=root
        / "tmp_results/deepseek-v3.2/Qwen_Qwen2.5-1.5B-Instruct-google/mt5-base-temp/2026-05-11-15-41-10-050459-fp32-0.15-1-paired_json-512/raw_data.json",
        help="run 输出的 raw_data.json",
    )
    p.add_argument(
        "--new-data",
        type=Path,
        default=root / "new_data.json",
        help="原始配对语料",
    )
    args = p.parse_args()

    with open(args.args_json, encoding="utf-8") as f:
        cfg = json.load(f)
    subkey = str(cfg.get("subkey", "")).strip()
    if not subkey:
        print("args.json 中 subkey 为空，无法按 model 校验。")
        return

    with open(args.raw_data, encoding="utf-8") as f:
        raw = json.load(f)
    sampled_list = raw.get("sampled") or []
    n = len(sampled_list)

    with open(args.new_data, encoding="utf-8") as f:
        records = json.load(f)

    # 每个 model -> 规范化全文集合（仅非 human）
    by_model: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        m = str(rec.get("model", "")).strip()
        if m.lower() == "h":
            continue
        by_model[m].add(norm(str(rec.get("content", ""))))

    if subkey not in by_model:
        print(f"警告: new_data.json 中没有任何 model=={subkey!r} 的记录（strip 后精确匹配）。")

    subkey_set = by_model.get(subkey, set())
    other_union: set[str] = set()
    for m, sset in by_model.items():
        if m != subkey:
            other_union |= sset

    exact_subkey = 0
    exact_other_only = 0
    exact_both = 0
    no_exact = []

    for i, s in enumerate(sampled_list):
        sn = norm(s)
        in_s = sn in subkey_set
        in_o = sn in other_union
        if in_s and not in_o:
            exact_subkey += 1
        elif in_s and in_o:
            exact_both += 1
        elif in_o and not in_s:
            exact_other_only += 1
            no_exact.append((i, "exact_match_wrong_model_only", sn[:80]))
        else:
            no_exact.append((i, "no_exact_norm_match", sn[:80]))

    # 软匹配：仅对仍未解释的索引，在 subkey 行中找 norm(content) 是否以 sn 为前缀或 sn 为其前缀（截断导致）
    soft_ok = 0
    still_bad: list[tuple[int, str]] = []
    for idx, kind, preview in list(no_exact):
        if kind == "exact_match_wrong_model_only":
            still_bad.append((idx, kind))
            continue
        sn = norm(sampled_list[idx])
        hit = False
        for rec in records:
            m = str(rec.get("model", "")).strip()
            if m != subkey:
                continue
            cn = norm(str(rec.get("content", "")))
            if not cn:
                continue
            if sn == cn or sn.startswith(cn) or cn.startswith(sn):
                hit = True
                break
            # token 截断常见：较短一方是较长一方的前缀（字面上）
            shorter, longer = (sn, cn) if len(sn) <= len(cn) else (cn, sn)
            if len(shorter) >= 200 and longer.startswith(shorter[:200]):
                hit = True
                break
        if hit:
            soft_ok += 1
        else:
            still_bad.append((idx, "no_match_even_soft"))

    print("=== temp.py 校验结果 ===")
    print(f"args.subkey: {subkey!r}")
    print(f"raw_data sampled 条数: {n}")
    print(f"new_data 中 model==subkey 的去重全文数: {len(subkey_set)}")
    print()
    print(f"规范化后与 subkey 语料完全一致（且未与其它 model 全文重复）: {exact_subkey}")
    print(f"规范化后与 subkey 一致且与其它 model 某条全文也相同（歧义）: {exact_both}")
    print(f"规范化后只与其它 model 全文一致（异常）: {exact_other_only}")
    print(f"无精确规范化匹配、但软匹配命中 subkey 行: {soft_ok}")
    print(f"仍无法对应到 subkey 语料（可能截断/改写）: {len(still_bad)}")
    if still_bad:
        print("\n前 20 条问题索引预览:")
        for idx, kind in still_bad[:20]:
            prev = norm(sampled_list[idx])[:100].replace("\n", " ")
            print(f"  idx={idx} kind={kind} preview={prev!r}...")


if __name__ == "__main__":
    main()
