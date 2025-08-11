# evaluate.py

from __future__ import annotations

import argparse
from datetime import datetime
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd
from cvss import CVSS3

# ----------------------------------
# Constants & simple configuration
# ----------------------------------

RESPONSE_COL = "response"
GT_COL = "ground_truth"

DEFAULT_MODELS: List[str] = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "deepseek-v3",
    "gemini-25-flash",
    "llama-4-maverick",
]

DEFAULT_OUTPUT_FILE = "evaluation_results.csv"

# ----------------------------------
# Utility helpers
# ----------------------------------


def _require_cols(df: pd.DataFrame, cols: Sequence[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def _pair_rows(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> Iterator[Tuple[pd.Series, pd.Series, int]]:
    """
    Iterate pairwise over pred and ground-truth rows up to the min length,
    yielding (pred_row, gt_row, row_index_0_based).
    """
    n = min(len(pred_df), len(gt_df))
    for i in range(n):
        yield pred_df.iloc[i], gt_df.iloc[i], i


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    return str(x)


def _print_and_write(fh, line: str) -> None:
    print(line)
    fh.write(line + "\n")

# ----------------------------------
# Formatting / Parsing Functions
# ----------------------------------


def format_mcq(text: str | object) -> str:
    """
    Parse an MCQ answer letter from free text.
    Strategy:
      1) First/last token on a line like 'A)' or ending in 'B'
      2) Markdown **A**
      3) Fallback: return normalized single letter if text is exactly A-D (case-insensitive)
    """
    s = _safe_str(text)
    for line in s.strip().splitlines():
        line = line.strip()
        if re.match(r"^[A-D]\)", line):
            return line[0].upper()
        if re.search(r"([A-D])\s*$", line):
            # type: ignore
            return re.search(r"([A-D])\s*$", line).group(1).upper()
    m = re.search(r"\*\*([A-D])\*\*", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    s2 = s.strip().upper()
    if s2 in {"A", "B", "C", "D"}:
        return s2
    return s  # fall back to raw (will be treated as invalid)


def format_rcm(text: str | object) -> Tuple[str, bool]:
    """Extract the last CWE ID (e.g., CWE-79) from free text."""
    s = _safe_str(text)
    matches = re.findall(r"CWE-\d+", s, flags=re.IGNORECASE)
    return (matches[-1].upper(), True) if matches else (s, False)


def format_vsp(text: str | object) -> Tuple[str, bool]:
    """
    Extract a CVSS v3.x base vector from free text and normalize it to a bare vector:
    'AV:X/AC:Y/PR:Z/UI:W/S:U|C/C:.../I:.../A:...'.
    Returns (vector_without_prefix, True) on success; otherwise (original_text, False).
    """
    s = _safe_str(text)
    s = " ".join(s.strip().split())
    pattern = re.compile(
        r"(?:CVSS:3\.[01]\s*/\s*)?"
        r"AV\s*:\s*[NALP]\s*/\s*"
        r"AC\s*:\s*[LH]\s*/\s*"
        r"PR\s*:\s*[NLH]\s*/\s*"
        r"UI\s*:\s*[NR]\s*/\s*"
        r"S\s*:\s*[UC]\s*/\s*"
        r"C\s*:\s*[NLH]\s*/\s*"
        r"I\s*:\s*[NLH]\s*/\s*"
        r"A\s*:\s*[NLH]",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(s))
    if not matches:
        return (s, False)
    m = matches[-1].group(0)
    m = re.sub(r"(?i)^CVSS:3\.[01]\s*/\s*", "", m)
    m = re.sub(r"\s+", "", m.upper()).replace("//", "/")
    return (m, True)


_TECH_ID_RE = re.compile(r"(?i)\bT\d{4}(?:\.\d{3})?\b")


def format_ate(text: str | object) -> List[str]:
    """
    Parse ATT&CK technique IDs from free text and return **primary** IDs (drop sub-techniques).
    E.g., 'T1059.001' -> 'T1059'. Sorted unique list.
    """
    s = _safe_str(text)
    # Fix cases like "T 1059"
    s = re.sub(r"(?i)T\s+(?=\d{4})", "T", s)
    ids = {m.group(0).upper().split(".")[0] for m in _TECH_ID_RE.finditer(s)}
    return sorted(ids)


_CLEAN_ACTOR_RE = re.compile(r"[^a-z0-9+/_-]+")


def format_taa(text: str | object, *, mapping: Optional[Dict[str, str]] = None) -> Set[str]:
    """
    Normalize free-text actor labels into a canonical set.
    - Splits on commas, semicolons, slashes, pipes, 'and', '&'
    - Lowercases, strips, squashes space/punct
    - Optional synonym mapping (exact or space-removed key)
    """
    s = _safe_str(text).lower()
    for sep in [";", "|", "/", "&", " and "]:
        s = s.replace(sep, ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: Set[str] = set()
    for p in parts:
        p2 = _CLEAN_ACTOR_RE.sub(" ", p).strip()
        p2 = re.sub(r"\s+", " ", p2)
        if not p2:
            continue
        if mapping:
            key = p2
            key2 = p2.replace(" ", "")
            out.add(mapping.get(key) or mapping.get(key2) or key)
        else:
            out.add(p2)
    return out

# ----------------------------------
# Evaluation Functions
# ----------------------------------


def compute_mcq_accuracy(predictions_df: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    correct, total = 0, 0
    _require_cols(predictions_df, [RESPONSE_COL], "predictions_df")
    _require_cols(gt_df, [GT_COL], "gt_df")
    for pred_row, gt_row, _ in _pair_rows(predictions_df, gt_df):
        pred = format_mcq(pred_row[RESPONSE_COL])
        gt = _safe_str(gt_row[GT_COL]).strip().upper()
        if pred in {"A", "B", "C", "D"}:
            total += 1
            if pred == gt:
                correct += 1
    return (100.0 * correct / total) if total else 0.0


def compute_rcm_accuracy(predictions_df: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    correct, total = 0, 0
    _require_cols(predictions_df, [RESPONSE_COL], "predictions_df")
    _require_cols(gt_df, [GT_COL], "gt_df")
    for pred_row, gt_row, i in _pair_rows(predictions_df, gt_df):
        pred, ok = format_rcm(pred_row[RESPONSE_COL])
        if not ok:
            continue
        total += 1
        gt = _safe_str(gt_row[GT_COL]).strip().upper()
        if pred.upper() == gt:
            correct += 1
    return (100.0 * correct / total) if total else 0.0


def compute_vsp_mad(predictions_df: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    """
    Mean Absolute Deviation between predicted vs ground-truth **base scores**.
    Predictions may omit the 'CVSS:3.x/' prefix; ground truth may include it.
    """
    _require_cols(predictions_df, [RESPONSE_COL], "predictions_df")
    _require_cols(gt_df, [GT_COL], "gt_df")
    total, err_sum = 0, 0.0

    for pred_row, gt_row, i in _pair_rows(predictions_df, gt_df):
        pred_vec, ok = format_vsp(pred_row.get(RESPONSE_COL, ""))
        gt_raw = _safe_str(gt_row.get(GT_COL, "")).strip().upper()

        # Normalize GT to full vector with version prefix
        gt_vec_full = gt_raw if gt_raw.startswith(
            "CVSS:") else f"CVSS:3.1/{gt_raw}"

        if not ok:
            continue
        pred_vec_full = f"CVSS:3.1/{pred_vec}"
        try:
            pred_score = CVSS3(pred_vec_full).scores()[0]
            gt_score = CVSS3(gt_vec_full).scores()[0]
            err_sum += abs(pred_score - gt_score)
            total += 1
        except Exception as e:
            # Skip malformed vectors silently but continue
            continue
    return (err_sum / total) if total else 0.0


def compute_ate_microf1(predictions_df: pd.DataFrame, gt_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Micro-averaged precision/recall/F1 on ATT&CK technique IDs.
    """
    _require_cols(predictions_df, [RESPONSE_COL], "predictions_df")
    _require_cols(gt_df, [GT_COL], "gt_df")
    tp = fp = fn = 0
    for pred_row, gt_row, _ in _pair_rows(predictions_df, gt_df):
        P = set(format_ate(pred_row.get(RESPONSE_COL, "")))
        T = {t.strip().upper().split(".")[0] for t in _safe_str(
            gt_row.get(GT_COL, "")).split(",") if t.strip()}
        tp += len(P & T)
        fp += len(P - T)
        fn += len(T - P)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _is_alias(a1: str, a2: str, alias_dict: Dict[str, List[str]]) -> bool:
    """BFS over alias graph (case: inputs already lowercased)."""
    q: List[str] = [a1]
    seen: Set[str] = {a1}
    while q:
        cur = q.pop(0)
        if cur == a2:
            return True
        for nxt in alias_dict.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def _is_related(a1: str, a2: str, alias_dict: Dict[str, List[str]], related_dict: Dict[str, List[str]]) -> bool:
    """BFS over alias + related graph (case: inputs already lowercased)."""
    q: List[str] = [a1]
    seen: Set[str] = {a1}
    while q:
        cur = q.pop(0)
        if cur == a2:
            return True
        for nxt in alias_dict.get(cur, []) + related_dict.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def compute_taa_accuracy(
    predictions_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    alias_dict: Dict[str, List[str]],
    related_dict: Dict[str, List[str]],
) -> Tuple[float, float]:
    """
    Threat Actor Attribution:
    - 'Correct' if prediction is an alias of the ground truth.
    - 'Plausible' if prediction is related (through alias or related graph).
    Accuracy reported as percentages over all prediction rows.
    """
    _require_cols(predictions_df, [RESPONSE_COL], "predictions_df")
    _require_cols(gt_df, [GT_COL], "gt_df")

    # Normalize dictionaries to lowercase and make them symmetric for traversal
    alias_lc = {k.strip().lower(): [v.strip().lower()
                                    for v in vals] for k, vals in alias_dict.items()}
    related_lc = {k.strip().lower(): [v.strip().lower()
                                      for v in vals] for k, vals in related_dict.items()}
    # Symmetrize
    for k, vals in list(alias_lc.items()):
        for v in vals:
            alias_lc.setdefault(v, []).append(k)
    for k, vals in list(related_lc.items()):
        for v in vals:
            related_lc.setdefault(v, []).append(k)

    correct = plausible = 0
    total = 0

    for pred_row, gt_row, _ in _pair_rows(predictions_df, gt_df):
        pred = " ".join(
            _safe_str(pred_row[RESPONSE_COL]).split()).lower().strip()
        gt = " ".join(_safe_str(gt_row[GT_COL]).split()).lower().strip()
        if not pred or not gt:
            continue
        total += 1
        if _is_alias(gt, pred, alias_lc):
            correct += 1
        elif _is_related(gt, pred, alias_lc, related_lc):
            plausible += 1

    correct_acc = (100.0 * correct / total) if total else 0.0
    plausible_acc = (100.0 * (correct + plausible) / total) if total else 0.0
    return correct_acc, plausible_acc

# ----------------------------------
# Orchestration
# ----------------------------------


@dataclass
class TaskPaths:
    predictions: str
    ground_truth: str
    alias_dict: Optional[str] = None
    related_dict: Optional[str] = None


def _task_map(model_root: str) -> Dict[str, TaskPaths]:
    return {
        "mcq": TaskPaths(
            predictions=f"results_{model_root}/results_mcq_{model_root}.csv",
            ground_truth="data/mcq.csv",
        ),
        "rcm": TaskPaths(
            predictions=f"results_{model_root}/results_rcm_{model_root}.csv",
            ground_truth="data/rcm.csv",
        ),
        "vsp": TaskPaths(
            predictions=f"results_{model_root}/results_vsp_{model_root}.csv",
            ground_truth="data/vsp.csv",
        ),
        "ate": TaskPaths(
            predictions=f"results_{model_root}/results_ate_{model_root}.csv",
            ground_truth="data/ate.csv",
        ),
        "taa": TaskPaths(
            predictions=f"results_{model_root}/results_taa_{model_root}.csv",
            ground_truth="data/taa.csv",
            alias_dict="data/taa_alias_dict.pickle",
            related_dict="data/taa_related_dict.pickle",
        ),
    }


def _evaluate_task(task: str, paths: TaskPaths) -> Optional[str]:
    try:
        pred_df = pd.read_csv(paths.predictions)
        gt_df = pd.read_csv(paths.ground_truth)
    except FileNotFoundError as e:
        return f"Skipping task '{task}': missing file - {e}"

    if task == "mcq":
        acc = compute_mcq_accuracy(pred_df, gt_df)
        return f"Accuracy: {acc:.2f}%"
    if task == "rcm":
        acc = compute_rcm_accuracy(pred_df, gt_df)
        return f"Accuracy: {acc:.2f}%"
    if task == "vsp":
        mad = compute_vsp_mad(pred_df, gt_df)
        return f"MAD: {mad:.2f}"
    if task == "ate":
        p, r, f1 = compute_ate_microf1(pred_df, gt_df)
        return f"Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}"
    # if task == "taa":
    #     try:
    #         assert paths.alias_dict and paths.related_dict
    #         with open(paths.alias_dict, "rb") as fh:
    #             alias_dict = pickle.load(fh)
    #         with open(paths.related_dict, "rb") as fh:
    #             related_dict = pickle.load(fh)
    #     except FileNotFoundError as e:
    #         return f"Skipping task 'taa': missing dictionary file - {e}"
    #     correct_acc, plausible_acc = compute_taa_accuracy(pred_df, gt_df, alias_dict, related_dict)
    #     return f"Correct Accuracy: {correct_acc:.2f}%, Plausible Accuracy: {plausible_acc:.2f}%"
    return f"Unknown task '{task}'"


START_BORDER = "=" * 60
MODEL_BORDER = "-" * 60
END_BORDER = "=" * 60


def evaluate_models(
    models: Sequence[str],
    tasks: Optional[Sequence[str]] = None,
    output_file: str = DEFAULT_OUTPUT_FILE,
) -> None:
    selected_tasks = set((t.strip().lower() for t in tasks)) if tasks else None

    with open(output_file, "a") as fh:
        # Start section
        lines = [
            START_BORDER,
            f"EVALUATION START",
            f"Models: {', '.join(models)}",
            f"Timestamp: {datetime.now().isoformat()}",
            START_BORDER,
        ]
        _print_and_write(fh, "\n".join(lines) + "\n")

        for model_root in models:
            # Model header
            _print_and_write(fh, f"\n{MODEL_BORDER}")
            _print_and_write(
                fh, f"Running evaluations for model: {model_root}")
            _print_and_write(fh, f"{MODEL_BORDER}\n")

            task_map = _task_map(model_root)
            results: Dict[str, str] = {}

            for task, paths in task_map.items():
                if selected_tasks and task not in selected_tasks:
                    continue
                msg = _evaluate_task(task, paths)
                if msg and not msg.startswith("Skipping"):
                    results[task] = msg
                _print_and_write(
                    fh, f"{task.upper()}: {msg}" if msg else f"{task.upper()}: (no result)"
                )

            if not results:
                _print_and_write(
                    fh, "No evaluations were run. Check file paths / task filters."
                )

        # End section
        lines = [
            END_BORDER,
            "EVALUATION COMPLETE",
            f"End Timestamp: {datetime.now().isoformat()}",
            END_BORDER,
        ]
        _print_and_write(fh, "\n" + "\n".join(lines) + "\n\n")

# ----------------------------------
# CLI entry
# ----------------------------------


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CTI-Bench evaluation runner")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                   help="Comma-separated list of model roots (default: built-in set)")
    p.add_argument("--tasks", type=str, default="",
                   help="Optional comma-separated tasks subset: mcq,rcm,vsp,ate,taa")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE,
                   help="Output CSV file to append results to")
    return p.parse_args()


def main() -> None:
    args = _parse_cli()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip().lower() for t in args.tasks.split(",")
             if t.strip()] if args.tasks else None
    evaluate_models(models, tasks=tasks, output_file=args.output)


if __name__ == "__main__":
    main()
