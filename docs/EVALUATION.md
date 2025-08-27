# Official Evaluation and Submission

This repository implements the official preliminary scoring you provided and includes tools to create properly named CSV submissions.

## CSV Formats (per task)
- KIS: `video_id,frame_idx`
- VQA: `video_id,frame_idx,answer`
- TRAKE: `video_id,frame1,frame2,...,frameN` (generator not provided here)

Rules
- Up to 100 lines per query (no header).
- One file per query, named `submissions/{query_id}.csv`.

## Generate Submissions (KIS/VQA)

Option A — Single query
```bash
# KIS
python src/retrieval/use.py --query "your search" --query_id q123

# VQA (3 columns)
python src/retrieval/use.py --query "question text" --task vqa --answer "màu xanh" --query_id q_vqa_01
```

Option B — Batch from JSON spec
```json
[
  {"query_id":"q1","task":"kis","query":"a person opening a laptop"},
  {"query_id":"q2","task":"vqa","query":"what color is the cup?","answer":"màu xanh"}
]
```

```bash
python scripts/make_submission.py --spec queries.json --index_dir ./artifacts
```

Artifacts Bundle and Reranker URLs (optional)
```bash
python scripts/make_submission.py --spec queries.json \
  --bundle_url https://host/artifacts_bundle.tar.gz \
  --model_url https://host/reranker.joblib
```

## Evaluate Locally

Ground truth JSON example
```json
[
  {"query_id":"q1","task":"kis","video_id":"L21_V001","span":[500,510]},
  {"query_id":"q2","task":"vqa","video_id":"L22_V002","span":[800,900],"answer":"màu xanh"}
]
```

Run evaluation
```bash
python eval/evaluate.py --gt ground_truth.json --pred_dir submissions --task kis
python eval/evaluate.py --gt ground_truth.json --pred_dir submissions --task vqa --normalize_answer
# TRAKE is supported for evaluation if you provide per-query CSVs in that format
```

Scoring definition
- R-Score: as per your spec per task (KIS/VQA=0/1 exactness; TRAKE=proportion of moments in spans). 
- Final Score: mean of best R-Score at Top-1, Top-5, Top-20, Top-50, Top-100 (max over first k predictions at each k).

Notes
- Ground truth spans are inclusive intervals: a frame t is correct if `span_start <= t <= span_end`.
- For epsilon-based centers, convert to intervals: `[center-eps, center+eps]` (KIS/VQA), or per-moment lists for TRAKE.
