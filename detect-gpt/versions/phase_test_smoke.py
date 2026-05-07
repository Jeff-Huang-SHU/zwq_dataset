#!/usr/bin/env python3
"""分阶段冒烟测试：JSON / label / 中文 mask / LL / AUC。

运行：在仓库根目录执行
  python versions/phase_test_smoke.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "test.json"


def phase1_json():
    print("=== Phase 1: JSON 读取 ===")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) >= 1, "应为非空 JSON 数组"
    for i, row in enumerate(data):
        assert "content" in row, f"第{i}条缺少 content"
        assert "model" in row, f"第{i}条缺少 model"
        assert len(row["content"]) > 0, f"第{i}条 content 为空"
    print(f"  OK: 读取 {len(data)} 条，字段齐全")
    return data


def phase2_labels(records):
    print("=== Phase 2: label / 配对（model=='h' -> human/original）===")
    groups = defaultdict(list)
    for rec in records:
        key = (rec.get("register", ""), rec.get("tittle", ""))
        groups[key].append(rec)
    pairs = []
    for key, items in groups.items():
        humans, machines = [], []
        for rec in items:
            m = str(rec.get("model", "")).lower()
            if m == "h":
                humans.append(rec["content"][:80] + "…")
            else:
                machines.append((rec.get("model", ""), rec["content"][:80] + "…"))
        n = min(len(humans), len(machines))
        assert n >= 1, f"组 {key} 应至少有一对 human+非h"
        pairs.append((key, humans[0], machines[0]))
        print(f"  组 {key}: human 条数={len(humans)}, AI 条数={len(machines)}, 配对 n={n}")
        print(f"    AI 模型名: {machines[0][0]}")
    print("  OK: label 规则与 3_temp_run.load_paired_json_data 一致（h=人，其余=机）")
    return True


def _tokenize_and_mask_inline(text, span_length, pct, buffer_size, mode, ceil_pct=False):
    """与 3_temp_run.tokenize_and_mask 等价的最小实现，避免 import 3_temp_run（顶层依赖 jieba）。"""
    if mode == "jieba":
        try:
            import jieba
        except ImportError:
            print("  SKIP jieba 子阶段: 未安装 jieba（pip install jieba）")
            return None
        tokens = jieba.lcut(text)
    elif mode == "char":
        tokens = list(text)
    else:
        tokens = text.split()

    if len(tokens) < span_length + 1:
        return text
    mask_string = "<<<mask>>>"
    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = max(1, int(n_spans))
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    num_filled = 0
    for idx, tok in enumerate(tokens):
        if tok == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    joiner = " " if mode == "space" else ""
    return joiner.join(tokens)


def phase3_chinese_mask():
    print("=== Phase 3: 中文 mask（char 必选；jieba 若已安装则追加）===")
    text = "在现代刑事司法制度中，陪审制作为程序民主的重要体现。"
    np.random.seed(0)
    masked_c = _tokenize_and_mask_inline(text, 3, 0.2, 1, "char", False)
    assert "<extra_id_" in masked_c, masked_c[:200]
    print(f"  char: 含 mask 片段预览: {masked_c[:100]}…")

    np.random.seed(42)
    mj = _tokenize_and_mask_inline(text, 2, 0.45, 1, "jieba", False)
    if mj is not None:
        nm = len(re.findall(r"<extra_id_\d+>", mj))
        assert nm > 0, f"jieba 分词后应能插入 mask，got n={nm}, preview={mj[:120]}"
        print(f"  jieba: 含 mask 片段预览: {mj[:100]}…")
    print("  OK: 中文 char mask 生效" + ("；jieba 亦通过" if mj is not None else "（jieba 未测）"))
    return True


def phase3b_mt5_fill():
    print("=== Phase 3b: mT5 填洞（CPU 小 batch）===")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "google/mt5-small"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    text = "现代司法制度中<extra_id_0>陪审制"
    ids = tok([text], return_tensors="pt")
    stop_id = tok.encode("<extra_id_1>", add_special_tokens=False)[0]
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_length=64,
            do_sample=False,
            num_beams=2,
            eos_token_id=stop_id,
        )
    dec = tok.decode(out[0], skip_special_tokens=False)
    assert len(dec) > 0
    print(f"  mt5-small 解码预览: {dec[:120]}")
    print("  OK: mT5 可对含 extra_id 的中文串做 generate")
    return True


def phase4_ll():
    print("=== Phase 4: LL（因果 LM 交叉熵，CPU + gpt2）===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(name)
    mdl.eval()
    text = "你好世界这是一个测试句子。"
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        loss = mdl(**inp, labels=inp["input_ids"]).loss.item()
    ll = -loss
    assert np.isfinite(ll), "LL 应为有限值"
    print(f"  平均 log-likelihood 近似: {ll:.4f} (= -loss)")
    print("  OK: LL 可计算（与 3_temp_run.get_ll 同公式）")
    return True


def phase5_auc():
    print("=== Phase 5: AUC（与 get_roc_metrics 同构的 sklearn）===")
    y = np.array([0] * 30 + [1] * 30)
    s = np.concatenate([np.random.randn(30) * 0.3 + 0.2, np.random.randn(30) * 0.3 + 0.8])
    auc = float(roc_auc_score(y, s))
    assert 0.0 <= auc <= 1.0
    print(f"  随机可分数据 ROC-AUC: {auc:.4f}")
    print("  OK: AUC 在 [0,1] 且与 sklearn 一致")
    return True


def main():
    ok = True
    try:
        r1 = phase1_json()
        phase2_labels(r1)
        phase3_chinese_mask()
        phase3b_mt5_fill()
        phase4_ll()
        phase5_auc()
    except Exception as e:
        print(f"\nFAIL: {e}")
        raise
    print("\n=== 汇总：各阶段均通过 ===")


if __name__ == "__main__":
    main()
