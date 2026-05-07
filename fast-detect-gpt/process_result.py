import json
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)


def load_scores(json_path: str):
    """
    从 FastDetectGPT 结果 JSON 中读取 real / samples 分数。

    real:    人类文本分数
    samples: AI 文本分数
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


def build_labels_and_scores(real_scores, sample_scores):
    """
    构造二分类标签和预测分数。

    label = 0: human / real / original
    label = 1: AI / samples / sampled

    默认假设 criterion 越大，越像 AI。
    """
    y_true = np.array(
        [0] * len(real_scores) +
        [1] * len(sample_scores),
        dtype=int
    )

    y_score = np.concatenate([real_scores, sample_scores])

    return y_true, y_score


def find_best_f1_threshold(y_true, y_score):
    """
    在 PR 曲线上寻找 F1 最大的阈值。
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(
        y_true,
        y_score
    )

    # precision_arr 和 recall_arr 比 thresholds 多一个点
    precision_use = precision_arr[:-1]
    recall_use = recall_arr[:-1]

    f1_arr = 2 * precision_use * recall_use / (
        precision_use + recall_use + 1e-12
    )

    best_idx = int(np.argmax(f1_arr))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_arr[best_idx])

    return best_threshold, best_f1


def compute_metrics(y_true, y_score, threshold):
    """
    根据 threshold 计算 accuracy / precision / recall / f1 / confusion matrix。
    """
    y_pred = (y_score >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(metrics, n_real, n_samples, json_roc_auc=None, json_pr_auc=None):
    total = n_real + n_samples

    print("=" * 60)
    print("FastDetectGPT result metrics")
    print("=" * 60)
    print(f"human / real 数量:   {n_real}")
    print(f"AI / samples 数量:   {n_samples}")
    print(f"总文本数:            {total}")
    print(f"paired 样本数:       {min(n_real, n_samples)}")
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


def save_metrics(metrics, output_path, n_real, n_samples, threshold_mode):
    output = {
        "n_real": int(n_real),
        "n_samples": int(n_samples),
        "n_total": int(n_real + n_samples),
        "n_pairs": int(min(n_real, n_samples)),
        "threshold_mode": threshold_mode,
        **metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"指标已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy, precision, recall, F1, ROC-AUC and PR-AUC from FastDetectGPT result JSON."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="FastDetectGPT 输出的结果 JSON 路径"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="手动指定阈值。例如 --threshold 0。如果不指定，则使用 F1 最优阈值。"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：保存统计指标到 JSON 文件"
    )

    args = parser.parse_args()

    data, real_scores, sample_scores = load_scores(args.input)
    y_true, y_score = build_labels_and_scores(real_scores, sample_scores)

    if args.threshold is None:
        threshold, _ = find_best_f1_threshold(y_true, y_score)
        threshold_mode = "best_f1"
    else:
        threshold = args.threshold
        threshold_mode = "manual"

    metrics = compute_metrics(y_true, y_score, threshold)

    json_roc_auc = None
    json_pr_auc = None

    if "metrics" in data and "roc_auc" in data["metrics"]:
        json_roc_auc = float(data["metrics"]["roc_auc"])

    if "pr_metrics" in data and "pr_auc" in data["pr_metrics"]:
        json_pr_auc = float(data["pr_metrics"]["pr_auc"])

    print_metrics(
        metrics=metrics,
        n_real=len(real_scores),
        n_samples=len(sample_scores),
        json_roc_auc=json_roc_auc,
        json_pr_auc=json_pr_auc,
    )

    if args.output is not None:
        save_metrics(
            metrics=metrics,
            output_path=args.output,
            n_real=len(real_scores),
            n_samples=len(sample_scores),
            threshold_mode=threshold_mode,
        )


if __name__ == "__main__":
    main()
"""
python process_result.py \
  --input exp_zwq/results/zwq_qwen2.5-1.5b.sampling_discrepancy_analytic.json
"""