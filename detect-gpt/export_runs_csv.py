"""
将单次 run 目录下 metrics/*_summary.json 汇总行追加到
<detect-gpt 父目录>/detect_gpt_runs.csv（存在则追加，不存在则创建）。

也可命令行执行：python export_runs_csv.py --scan-all
扫描 detect-gpt/results/**/metrics/ 下所有 perturbation_*_summary.json，跳过已在 CSV 中的重复键。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parent
_CSV_PATH = _REPO_ROOT.parent / "detect_gpt_runs.csv"

HEADERS = [
    "category",
    "subkey",
    "base_model",
    "maskfill_model",
    "n",
    "n_neg",
    "n_perturbation",
    "accuracy",
    "f1",
    "roc_auc",
    "precision",
    "recall",
]

_SUMMARY_RE = re.compile(r"^perturbation_(\d+)_(d|z)_summary\.json$")


def _row_key(row: Dict[str, Any]) -> Tuple:
    """用于去重：同一次 run、同一扰动次数、同一 d/z。"""
    return (
        row["category"],
        row["subkey"],
        row["base_model"],
        row["maskfill_model"],
        str(row["n_perturbation"]),
        str(row["n"]),
        str(row["n_neg"]),
    )


def _read_existing_keys(csv_path: Path) -> Set[Tuple]:
    if not csv_path.is_file():
        return set()
    keys: Set[Tuple] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return set()
        for r in reader:
            try:
                keys.add(
                    (
                        r.get("category", ""),
                        r.get("subkey", ""),
                        r.get("base_model", ""),
                        r.get("maskfill_model", ""),
                        str(r.get("n_perturbation", "")),
                        str(r.get("n", "")),
                        str(r.get("n_neg", "")),
                    )
                )
            except Exception:
                continue
    return keys


def _rows_from_run_dir(run_dir: Path) -> List[Dict[str, Any]]:
    run_dir = run_dir.resolve()
    metrics_dir = run_dir / "metrics"
    args_path = run_dir / "args.json"
    if not metrics_dir.is_dir() or not args_path.is_file():
        return []

    with open(args_path, encoding="utf-8") as f:
        args = json.load(f)

    subkey = str(args.get("subkey", "") or "")
    base_model = str(args.get("base_model_name", "") or "")
    maskfill_model = str(args.get("mask_filling_model_name", "") or "")

    rows: List[Dict[str, Any]] = []
    for p in sorted(metrics_dir.iterdir()):
        if not p.is_file():
            continue
        m = _SUMMARY_RE.match(p.name)
        if not m:
            continue
        n_perturbation = int(m.group(1))
        mode = m.group(2)
        category = f"perturbation_{mode}"

        with open(p, encoding="utf-8") as f:
            summary = json.load(f)

        n_total = int(summary.get("n_total", 0))
        n_real = int(summary.get("n_real", 0))
        n_samples = int(summary.get("n_samples", 0))

        rows.append(
            {
                "category": category,
                "subkey": subkey,
                "base_model": base_model,
                "maskfill_model": maskfill_model,
                "n": n_total,
                "n_neg": n_real,
                "n_perturbation": n_perturbation,
                "accuracy": float(summary.get("accuracy", 0.0)),
                "f1": float(summary.get("f1", 0.0)),
                "roc_auc": float(summary.get("roc_auc", 0.0)),
                "precision": float(summary.get("precision", 0.0)),
                "recall": float(summary.get("recall", 0.0)),
            }
        )
    return rows


def append_run_dir_to_csv(run_dir: str | Path, *, csv_path: Path | None = None) -> int:
    """
    将 run_dir（含 args.json 与 metrics/）下的 perturbation summary 追加到 CSV。
    返回新写入的行数（去重后）。
    """
    path = Path(run_dir)
    out_csv = csv_path or _CSV_PATH
    rows = _rows_from_run_dir(path)
    if not rows:
        return 0

    existing = _read_existing_keys(out_csv)
    new_rows: List[Dict[str, Any]] = []
    for r in rows:
        if _row_key(r) in existing:
            continue
        new_rows.append(r)
        existing.add(_row_key(r))

    if not new_rows:
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.is_file()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in new_rows:
            w.writerow(
                {
                    "category": r["category"],
                    "subkey": r["subkey"],
                    "base_model": r["base_model"],
                    "maskfill_model": r["maskfill_model"],
                    "n": r["n"],
                    "n_neg": r["n_neg"],
                    "n_perturbation": r["n_perturbation"],
                    "accuracy": f"{r['accuracy']:.10g}",
                    "f1": f"{r['f1']:.10g}",
                    "roc_auc": f"{r['roc_auc']:.10g}",
                    "precision": f"{r['precision']:.10g}",
                    "recall": f"{r['recall']:.10g}",
                }
            )
    print(f"Appended {len(new_rows)} row(s) to {out_csv}")
    return len(new_rows)


def iter_run_dirs_with_metrics(results_root: Path) -> Iterable[Path]:
    results_root = results_root.resolve()
    if not results_root.is_dir():
        return
    for metrics_dir in sorted(results_root.rglob("metrics")):
        if not metrics_dir.is_dir():
            continue
        run_dir = metrics_dir.parent
        if (run_dir / "args.json").is_file():
            yield run_dir


def scan_all_results_append(
    results_root: Path | None = None, *, csv_path: Path | None = None
) -> int:
    """扫描 results 下所有含 metrics 的 run，追加尚未出现在 CSV 中的行。"""
    root = results_root or (_REPO_ROOT / "results")
    total = 0
    seen_run: Set[Path] = set()
    for run_dir in iter_run_dirs_with_metrics(root):
        if run_dir in seen_run:
            continue
        seen_run.add(run_dir)
        total += append_run_dir_to_csv(run_dir, csv_path=csv_path)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Append DetectGPT metrics rows to detect_gpt_runs.csv")
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="扫描整个 detect-gpt/results 下所有 run 并追加（去重）",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="仅处理该次实验目录（含 args.json 与 metrics/）",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=f"CSV 路径（默认 {_CSV_PATH}）",
    )
    args = parser.parse_args()
    csv_path = args.csv or _CSV_PATH
    if args.scan_all:
        n = scan_all_results_append(csv_path=csv_path)
        print(f"scan-all done, total new rows appended: {n}")
    elif args.run_dir:
        n = append_run_dir_to_csv(args.run_dir, csv_path=csv_path)
        print(f"run-dir done, new rows: {n}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
