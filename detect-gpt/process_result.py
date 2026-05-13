"""
从 DetectGPT（run.py）输出的结果 JSON 中读取 predictions，计算
accuracy、f1、roc_auc、precision、recall（及可选 pr_auc）。

输入/输出约定参考 fast-detect-gpt/process_result.py：
  --input  结果 JSON（须含 predictions.real / predictions.samples）
  --threshold  可选，手动阈值；默认在 PR 曲线上选 F1 最优阈值
  --output  可选，将指标写入 JSON

用法示例:
  python process_result.py \\
    --input results/.../perturbation_1_d_results.json \\
    --output metrics_summary.json
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_scores(json_path: str) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    从 DetectGPT 结果 JSON 中读取 real / samples 分数。

    real:    人类 / original
    samples: 机器 / sampled
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "predictions" not in data:
        raise KeyError("JSON 文件中没有 predictions 字段")

    predictions = data["predictions"]

    if "real" not in predictions or "samples" not in predictions:
        raise KeyError("predictions 中必须包含 real 和 samples 字段")

    real_scores = np.array(predictions["real"], dtype=float)
    sample_scores = np.array(predictions["samples"], dtype=float)

    if len(real_scores) == 0 or len(sample_scores) == 0:
        raise ValueError("real 或 samples 为空，无法计算指标")

    return data, real_scores, sample_scores


def build_labels_and_scores(
    real_scores: np.ndarray, sample_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    二分类标签与预测分数（与 run.py / fast-detect-gpt 一致）。

    label = 0: human / real / original
    label = 1: AI / samples / sampled

    默认后续假设：分数越高越像 AI（正类）。
    """
    y_true = np.array(
        [0] * len(real_scores) + [1] * len(sample_scores), dtype=int
    )
    y_score = np.concatenate([real_scores, sample_scores])
    return y_true, y_score


def maybe_flip_scores(y_true: np.ndarray, y_score: np.ndarray, auto_flip: bool) -> Tuple[np.ndarray, bool]:
    """
    若 roc_auc < 0.5，对分数取反，使「高分 = 更像 AI」，
    便于与 F1 最优阈值、accuracy 等解释一致（DetectGPT 各 criterion 方向不一）。
    """
    if not auto_flip:
        return y_score, False
    try:
        auc_raw = roc_auc_score(y_true, y_score)
    except ValueError:
        return y_score, False
    if np.isnan(auc_raw) or auc_raw >= 0.5:
        return y_score, False
    return -y_score, True


def find_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """在 PR 曲线上寻找 F1 最大的阈值。"""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_score)

    precision_use = precision_arr[:-1]
    recall_use = recall_arr[:-1]

    f1_arr = 2 * precision_use * recall_use / (precision_use + recall_use + 1e-12)

    best_idx = int(np.argmax(f1_arr))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_arr[best_idx])

    return best_threshold, best_f1


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """根据 threshold 计算 accuracy / precision / recall / f1 及 ROC/PR AUC。"""
    y_pred = (y_score >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_result_metrics(
    json_path: str,
    threshold: Optional[float] = None,
    *,
    auto_flip: bool = True,
) -> Dict[str, Any]:
    """
    根据 DetectGPT 结果 JSON 计算各项指标（供脚本或其它模块调用）。

    Returns
    -------
    dict
        含 threshold、accuracy、precision、recall、f1、roc_auc、pr_auc、
        混淆矩阵项、n_real、n_samples、score_flipped、threshold_mode 等。
    """
    data, real_scores, sample_scores = load_scores(json_path)
    y_true, y_score = build_labels_and_scores(real_scores, sample_scores)
    y_score, flipped = maybe_flip_scores(y_true, y_score, auto_flip)

    if threshold is None:
        thr, _ = find_best_f1_threshold(y_true, y_score)
        threshold_mode = "best_f1"
    else:
        thr = float(threshold)
        threshold_mode = "manual"

    metrics = compute_metrics(y_true, y_score, thr)

    out: Dict[str, Any] = {
        "input": json_path,
        "n_real": int(len(real_scores)),
        "n_samples": int(len(sample_scores)),
        "n_total": int(len(real_scores) + len(sample_scores)),
        "n_pairs": int(min(len(real_scores), len(sample_scores))),
        "threshold_mode": threshold_mode,
        "score_flipped": flipped,
        **metrics,
    }

    if "metrics" in data and "roc_auc" in data["metrics"]:
        out["json_roc_auc"] = float(data["metrics"]["roc_auc"])
    if "pr_metrics" in data and "pr_auc" in data["pr_metrics"]:
        out["json_pr_auc"] = float(data["pr_metrics"]["pr_auc"])

    return out


def print_metrics(
    metrics: Dict[str, Any],
    n_real: int,
    n_samples: int,
    json_roc_auc: Optional[float] = None,
    json_pr_auc: Optional[float] = None,
) -> None:
    total = n_real + n_samples

    print("=" * 60)
    print("DetectGPT result metrics")
    print("=" * 60)
    print(f"human / real 数量:   {n_real}")
    print(f"AI / samples 数量:   {n_samples}")
    print(f"总文本数:            {total}")
    print(f"paired 样本数:       {min(n_real, n_samples)}")
    if metrics.get("score_flipped"):
        print("分数已自动取反 (roc_auc<0.5 → 高分=AI)")
    print("-" * 60)
    print(f"threshold:           {metrics['threshold']:.6f}")
    print(f"accuracy:            {metrics['accuracy']:.6f}")
    print(f"precision:           {metrics['precision']:.6f}")
    print(f"recall:              {metrics['recall']:.6f}")
    print(f"f1:                  {metrics['f1']:.6f}")
    print(f"roc_auc:             {metrics['roc_auc']:.6f}")
    print(f"pr_auc:              {metrics['pr_auc']:.6f}")
    print("-" * 60)
    print(f"TN:                  {metrics['tn']}")
    print(f"FP:                  {metrics['fp']}")
    print(f"FN:                  {metrics['fn']}")
    print(f"TP:                  {metrics['tp']}")

    if json_roc_auc is not None or json_pr_auc is not None:
        print("-" * 60)
        if json_roc_auc is not None:
            print(f"JSON 中原始 roc_auc: {json_roc_auc:.6f}")
        if json_pr_auc is not None:
            print(f"JSON 中原始 pr_auc:  {json_pr_auc:.6f}")

    print("=" * 60)


def save_metrics(
    payload: Dict[str, Any], output_path: str
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"指标已保存到: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从 DetectGPT 结果 JSON 计算 accuracy、f1、roc_auc、precision、recall（及 pr_auc）。"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="DetectGPT 输出的结果 JSON 路径（含 predictions）",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="手动指定阈值；不指定则使用 PR 曲线上 F1 最优阈值",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：将指标保存到 JSON 文件",
    )

    parser.add_argument(
        "--no_auto_flip",
        action="store_true",
        help="禁用分数自动取反（默认：若 roc_auc<0.5 则取反使高分=AI）",
    )

    args = parser.parse_args()

    result = compute_result_metrics(
        args.input,
        threshold=args.threshold,
        auto_flip=not args.no_auto_flip,
    )

    json_roc = result.get("json_roc_auc")
    json_pr = result.get("json_pr_auc")

    metrics_for_print = {
        k: result[k]
        for k in (
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "tn",
            "fp",
            "fn",
            "tp",
            "score_flipped",
        )
    }

    print_metrics(
        metrics=metrics_for_print,
        n_real=result["n_real"],
        n_samples=result["n_samples"],
        json_roc_auc=json_roc,
        json_pr_auc=json_pr,
    )

    if args.output is not None:
        save_metrics(result, args.output)


if __name__ == "__main__":
    main()
