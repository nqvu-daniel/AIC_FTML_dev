#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


TOP_KS = [1, 5, 20, 50, 100]


def normalize_text(s: str) -> str:
    return s.strip().casefold()


def read_gt(gt_path: Path) -> Dict[str, dict]:
    data = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    out = {}
    for obj in data:
        qid = str(obj["query_id"])
        out[qid] = obj
    return out


def read_predictions_for_query(csv_path: Path) -> List[List[str]]:
    if not csv_path.exists():
        return []
    rows: List[List[str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            rows.append(parts)
    return rows


def rscore_kis(gt: dict, pred: List[str]) -> float:
    # pred: [video_id, frame_idx]
    if len(pred) < 2:
        return 0.0
    try:
        vid = pred[0]
        frame = int(pred[1])
    except Exception:
        return 0.0
    if vid != gt["video_id"]:
        return 0.0
    s, e = gt["span"][0], gt["span"][1]
    return 1.0 if (s <= frame <= e) else 0.0


def rscore_vqa(gt: dict, pred: List[str], normalize_answer: bool = False) -> float:
    # pred: [video_id, frame_idx, answer]
    if len(pred) < 3:
        return 0.0
    try:
        vid = pred[0]
        frame = int(pred[1])
        ans = pred[2]
    except Exception:
        return 0.0
    if vid != gt["video_id"]:
        return 0.0
    s, e = gt["span"][0], gt["span"][1]
    if not (s <= frame <= e):
        return 0.0
    gans = gt["answer"]
    if normalize_answer:
        return 1.0 if normalize_text(ans) == normalize_text(gans) else 0.0
    return 1.0 if ans == gans else 0.0


def rscore_trake(gt: dict, pred: List[str]) -> float:
    # pred: [video_id, frame1, frame2, ..., frameN]
    if len(pred) < 2:
        return 0.0
    vid = pred[0]
    if vid != gt["video_id"]:
        return 0.0
    spans: List[Tuple[int, int]] = [(int(s), int(e)) for s, e in gt["spans"]]
    frames: List[int] = []
    for p in pred[1:]:
        try:
            frames.append(int(p))
        except Exception:
            frames.append(-10**9)  # invalid → definitely outside any span
    # Only compare up to N moments
    n = min(len(spans), len(frames))
    if n == 0:
        return 0.0
    hits = 0
    for j in range(n):
        s, e = spans[j]
        if s <= frames[j] <= e:
            hits += 1
    return hits / float(len(spans))  # denominator = N per spec


def compute_r_at_k(r_scores: List[float], k: int) -> float:
    if not r_scores:
        return 0.0
    cutoff = min(k, len(r_scores))
    return max(r_scores[:cutoff]) if cutoff > 0 else 0.0


def final_score(r_scores: List[float]) -> Tuple[Dict[int, float], float]:
    r_at = {k: compute_r_at_k(r_scores, k) for k in TOP_KS}
    mean = sum(r_at.values()) / float(len(TOP_KS))
    return r_at, mean


def evaluate(
    gt: Dict[str, dict],
    pred_dir: Path,
    task_hint: Optional[str] = None,
    normalize_answer: bool = False,
    max_candidates: int = 100,
):
    per_query = {}
    macro_final_sum = 0.0  # also serves as bundle total (sum of per-query Final Scores)
    qids = sorted(gt.keys())
    for qid in qids:
        g = gt[qid]
        task = task_hint or g.get("task")
        if task not in {"kis", "vqa", "trake"}:
            print(f"[WARN] Unknown task for {qid}; skipping", file=sys.stderr)
            continue
        csv_path = pred_dir / f"{qid}.csv"
        preds = read_predictions_for_query(csv_path)[:max_candidates]
        r_scores: List[float] = []
        for p in preds:
            if task == "kis":
                r_scores.append(rscore_kis(g, p))
            elif task == "vqa":
                r_scores.append(rscore_vqa(g, p, normalize_answer=normalize_answer))
            else:
                r_scores.append(rscore_trake(g, p))
        r_at, fscore = final_score(r_scores)
        per_query[qid] = {"R@k": r_at, "FinalScore": fscore, "num_candidates": len(preds)}
        macro_final_sum += fscore

    macro_avg = macro_final_sum / float(len(per_query) or 1)
    return per_query, macro_avg


def main():
    ap = argparse.ArgumentParser(description="Evaluate AIC preliminary scoring: R-Score and Mean of Top-k Final Score")
    ap.add_argument("--gt", type=Path, required=True, help="Path to ground truth JSON")
    ap.add_argument("--pred_dir", type=Path, required=True, help="Directory containing per-query CSVs named {query_id}.csv")
    ap.add_argument("--task", type=str, choices=["kis", "vqa", "trake"], default=None, help="Override task for all queries (else read from GT)")
    ap.add_argument("--normalize_answer", action="store_true", help="Normalize VQA answers before matching")
    ap.add_argument("--max_candidates", type=int, default=100, help="Max candidates per query (cap to 100)")
    args = ap.parse_args()

    gt = read_gt(args.gt)
    per_query, macro_avg = evaluate(
        gt=gt,
        pred_dir=args.pred_dir,
        task_hint=args.task,
        normalize_answer=args.normalize_answer,
        max_candidates=min(100, max(1, args.max_candidates)),
    )

    # Pretty print summary
    print("=== Per-query Results ===")
    for qid, res in per_query.items():
        r_at = res["R@k"]
        fs = res["FinalScore"]
        nc = res["num_candidates"]
        r_str = ", ".join([f"R@{k}={r_at[k]:.4f}" for k in TOP_KS])
        print(f"{qid}: {r_str} | Final={fs:.4f} | n={nc}")
    print("=========================")
    bundle_total = sum(res["FinalScore"] for res in per_query.values())
    print(f"Bundle total (sum of Final Scores): {bundle_total:.4f}")
    print(f"Macro-average Final Score: {macro_avg:.4f}")


if __name__ == "__main__":
    main()
