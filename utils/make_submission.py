#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def load_queries(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # allow single object with {queries:[...]}
        data = data.get("queries", [])
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array of query objects")
    # expected fields: query_id, task (kis|vqa), query, [answer]
    return data


def run_use_py(qid: str, task: str, query: str, answer: str | None, index_dir: Path, topk: int, dedup_radius: int, bundle_url: str | None, model_url: str | None, default_clip: bool = False, experimental: bool = False, exp_model: str | None = None, exp_pretrained: str | None = None):
    # Use new search interface
    cmd = [
        "python",
        "search.py",
        "--query",
        query,
        "--artifact_dir",
        str(index_dir),
        "--k",
        str(topk),
        "--output",
        f"submissions/{qid}.csv"
    ]
    # Add model options for new search interface
    if default_clip:
        cmd += ["--model_name", "ViT-B-32", "--pretrained", "openai"]
    elif experimental and exp_model:
        cmd += ["--model_name", exp_model]
        if exp_pretrained:
            cmd += ["--pretrained", exp_pretrained]
    
    # Note: VQA task and bundle_url options not yet implemented in new search interface
    # These would need to be added to the new search.py if needed

    print("→", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise SystemExit(res.returncode)
    return True


def main():
    ap = argparse.ArgumentParser(description="Build official submissions from a query spec JSON")
    ap.add_argument("--spec", type=Path, required=True, help="Path to JSON file: [{query_id,task,query,answer?}, ...]")
    ap.add_argument("--index_dir", type=Path, default=Path("./artifacts"))
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--dedup_radius", type=int, default=1)
    ap.add_argument("--bundle_url", type=str, default=None, help="Optional artifacts bundle URL")
    ap.add_argument("--model_url", type=str, default=None, help="Optional reranker model URL")
    ap.add_argument("--default-clip", action="store_true", help="Use default ViT-B-32 CLIP (512D) to match 512D indexes")
    ap.add_argument("--experimental", action="store_true", help="Enable experimental model selection (advanced backbones)")
    ap.add_argument("--exp-model", type=str, default=None, help="Experimental model name or preset key (see config.EXPERIMENTAL_PRESETS)")
    ap.add_argument("--exp-pretrained", type=str, default=None, help="Override pretrained tag for experimental model")
    args = ap.parse_args()

    subs_dir = Path("submissions")
    subs_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.spec)
    n_ok = 0
    for q in queries:
        qid = str(q.get("query_id"))
        task = (q.get("task") or "kis").lower()
        if task not in {"kis", "vqa"}:
            print(f"[WARN] Skipping query_id={qid}: unsupported task '{task}' (supported: kis, vqa)")
            continue
        query = q.get("query")
        if not query:
            print(f"[WARN] Skipping query_id={qid}: missing 'query'")
            continue
        answer = q.get("answer")
        run_use_py(
            qid=qid,
            task=task,
            query=query,
            answer=answer,
            index_dir=args.index_dir,
            topk=args.topk,
            dedup_radius=args.dedup_radius,
            bundle_url=args.bundle_url,
            model_url=args.model_url,
            default_clip=args.default_clip,
            experimental=args.experimental,
            exp_model=args.exp_model,
            exp_pretrained=args.exp_pretrained,
        )
        n_ok += 1

    print(f"[OK] Wrote {n_ok} submission file(s) to {subs_dir}/")


if __name__ == "__main__":
    main()
