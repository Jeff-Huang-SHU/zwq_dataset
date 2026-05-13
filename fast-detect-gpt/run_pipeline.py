#!/usr/bin/env python3
"""
端到端：convert_dataset（格式 A→B *.raw_data.json）→ fast_detect_gpt.py → 追加汇总 CSV 到 A_workspace。
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

# 仓库根（含 scripts/）
_REPO_ROOT = Path(__file__).resolve().parent
_A_WORKSPACE = _REPO_ROOT.parent
_DEFAULT_CSV = _A_WORKSPACE / "fast_detect_gpt_runs.csv"


def _append_csv_row(
    csv_path: Path,
    row: dict,
    fieldnames: list[str],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def run_fast_detect(repo_root: Path, argv: list[str]) -> None:
    cmd = [sys.executable, str(repo_root / "scripts" / "fast_detect_gpt.py")] + argv
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="生成 raw_data.json 并运行 Fast-DetectGPT，汇总指标到 A_workspace CSV。"
    )
    # convert_dataset
    p.add_argument("--input", type=str, required=True, help="格式 A 输入 JSON")
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="raw_data 输出根目录（与 scripts/convert_dataset.py --output 一致）",
    )
    p.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neo-2.7B")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--subkey", type=str, default="")
    p.add_argument(
        "--cache_dir",
        type=str,
        default=str(_A_WORKSPACE / "cache"),
        help="HF 缓存（默认：上级 A_workspace/cache）",
    )
    # fast_detect_gpt
    p.add_argument("--dataset", type=str, default="mydata")
    p.add_argument("--sampling_model_name", type=str, default="gpt-neo-2.7B")
    p.add_argument("--scoring_model_name", type=str, default="gpt-neo-2.7B")
    p.add_argument("--discrepancy_analytic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Fast-DetectGPT 结果前缀所在目录（将写入 sampling_discrepancy*.json）",
    )
    p.add_argument(
        "--output_basename",
        type=str,
        default=None,
        help="结果文件前缀 basename（默认与 raw_data 主文件名 stem 一致）",
    )
    # CSV
    p.add_argument("--category", type=str, default="")
    p.add_argument(
        "--csv_path",
        type=str,
        default=str(_DEFAULT_CSV),
        help=f"汇总 CSV（默认 {_DEFAULT_CSV}）",
    )
    p.add_argument(
        "--skip_detect",
        action="store_true",
        help="仅转换数据，不运行 fast_detect_gpt",
    )

    args = p.parse_args()

    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    sys.path.insert(0, str(_REPO_ROOT))
    from convert_dataset import convert_format_a_to_b, dataset_file_prefix_from_raw_data_path

    raw_path = convert_format_a_to_b(
        input_path=args.input,
        output_dir=args.output,
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.cache_dir,
        max_tokens=args.max_tokens,
        subkey=args.subkey,
    )

    dataset_file = dataset_file_prefix_from_raw_data_path(raw_path)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.output_basename:
        out_base = results_dir / args.output_basename
    else:
        # raw 文件名为 xxx.raw_data.json → stem 为 xxx.raw_data，结果前缀用完整 stem 避免碰撞
        out_base = results_dir / raw_path.name[: -len(".raw_data.json")]

    name = "sampling_discrepancy_analytic" if args.discrepancy_analytic else "sampling_discrepancy"
    result_json = Path(f"{out_base}.{name}.json")

    if args.skip_detect:
        print(f"[skip_detect] raw_data: {raw_path}")
        print(f"[skip_detect] dataset_file 前缀: {dataset_file}")
        print(f"[skip_detect] 预期结果 JSON: {result_json}")
        return

    fd_argv = [
        "--dataset",
        args.dataset,
        "--dataset_file",
        dataset_file,
        "--output_file",
        str(out_base),
        "--sampling_model_name",
        args.sampling_model_name,
        "--scoring_model_name",
        args.scoring_model_name,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--cache_dir",
        args.cache_dir,
    ]
    if args.discrepancy_analytic:
        fd_argv.append("--discrepancy_analytic")

    run_fast_detect(_REPO_ROOT, fd_argv)

    if not result_json.is_file():
        raise FileNotFoundError(f"未找到评测输出: {result_json}")

    from process_result import (
        load_scores,
        build_labels_and_scores,
        find_best_f1_threshold,
        compute_metrics,
    )

    _, real_scores, sample_scores = load_scores(str(result_json))
    y_true, y_score = build_labels_and_scores(real_scores, sample_scores)
    threshold, _ = find_best_f1_threshold(y_true, y_score)
    m = compute_metrics(y_true, y_score, threshold)

    n_total = int(len(real_scores) + len(sample_scores))
    n_ai = int(len(sample_scores))

    csv_fields = [
        "category",
        "subkey",
        "sampling_model_name",
        "scoring_model_name",
        "n",
        "n_neg",
        "accuracy",
        "f1",
        "roc_auc",
        "precision",
        "recall",
    ]
    csv_row = {
        "category": args.category,
        "subkey": args.subkey.strip(),
        "sampling_model_name": args.sampling_model_name,
        "scoring_model_name": args.scoring_model_name,
        "n": n_total,
        "n_neg": n_ai,
        "accuracy": f"{m['accuracy']:.6f}",
        "f1": f"{m['f1']:.6f}",
        "roc_auc": f"{m['roc_auc']:.6f}",
        "precision": f"{m['precision']:.6f}",
        "recall": f"{m['recall']:.6f}",
    }
    _append_csv_row(Path(args.csv_path), csv_row, csv_fields)
    print(f"已追加 CSV 行: {args.csv_path}")


if __name__ == "__main__":
    main()
