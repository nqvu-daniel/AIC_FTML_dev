# Competition Dataset and Submissions (Host Spec)

This file describes only the official dataset as provided by the competition host and the required query and submission formats.

## Host Dataset (as distributed)

Top-level archives unpack into folders similar to:

```
Keyframes_L21/
  keyframes/
    L21_V001/
      001.jpg 002.jpg ...        # JPG/JPEG/PNG supported
Videos_L21_a/
  L21_V001.mp4 L21_V002.mp4 ...
map-keyframes-aic25-b1/
  map-keyframes/
    L21_V001.csv L21_V002.csv ...
media-info-aic25-b1/
  media-info/
    L21_V001.json L21_V002.json ...
objects-aic25-b1/
  objects/
    L26_V361/
      155.json 135.json ...      # Per-keyframe detections
clip-features-32-aic25-b1/
  clip-features-32/
    L21_V001.npy L21_V002.npy ...
```

Minimal file specs used by the task:
- `map-keyframes/{VIDEO_ID}.csv` (per video)
  - Columns: `n` (1‑indexed keyframe number), `pts_time` (seconds), `fps`, `frame_idx` (0‑indexed frame in the original video)
- `media-info/{VIDEO_ID}.json` (per video)
  - Includes `title`, `description`, `keywords`, and other metadata
- `objects/{VIDEO_ID}/{N:03d}.json` (per keyframe)
  - Includes `detection_class_entities`, `detection_scores`, `detection_boxes`
- `clip-features-32/{VIDEO_ID}.npy` (per video)
  - Numpy array of CLIP embeddings per keyframe (rows align with `n` in the CSV)

Note: The host may provide multiple Lxx collections (e.g., L21, L22, ...). Video IDs follow the pattern `L## _V###` (e.g., `L21_V001`).

## Queries (official intent)

The host provides a list of text queries with identifiers and task type:
- `query_id`: unique identifier for the query
- `task`: one of `kis`, `vqa`, or `trake`
- `query`: natural‑language description or question

Notes:
- For VQA, the answer is not given by the host; your submission must include the predicted answer text.
- TRAKE describes multi‑moment events; see submission format below for required output columns.

## Submission Outputs (required formats)

Create one CSV per query (no header, up to 100 lines), named exactly:
- `submissions/{query_id}.csv`

Each line is a ranked candidate from best to worst. Formats by task:
- KIS: `video_id,frame_idx`
- VQA: `video_id,frame_idx,answer`
- TRAKE: `video_id,frame1,frame2,...,frameN`

Conventions:
- `video_id` matches the dataset (e.g., `L21_V001`).
- `frame_idx` and `frame1..frameN` refer to original video frame indices.
- Do not include a header row; maximum 100 lines per query file.
- Scoring is performed by the competition host; this system only exports prediction CSVs and does not compute official scores.
