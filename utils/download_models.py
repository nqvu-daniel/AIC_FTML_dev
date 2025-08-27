import argparse
import os
from pathlib import Path
from urllib.request import urlopen, Request


def download_file(url: str, outfile: Path, chunk_size: int = 1 << 20):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:  # nosec - URL supplied by user
        total = int(r.headers.get("Content-Length", 0))
        read = 0
        tmp = outfile.with_suffix(outfile.suffix + ".tmp")
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if total:
                    pct = 100.0 * read / total
                    print(f"\rDownloading: {read/1e6:.1f}MB/{total/1e6:.1f}MB ({pct:.1f}%)", end="")
        if total:
            print("\nDone.")
        tmp.replace(outfile)


def main():
    ap = argparse.ArgumentParser(description="Download trained model artifacts")
    ap.add_argument("--model-url", required=True, help="URL to reranker.joblib")
    ap.add_argument(
        "--outfile",
        type=Path,
        default=Path("./artifacts/reranker.joblib"),
        help="Destination path for the downloaded model",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = ap.parse_args()

    if args.outfile.exists() and not args.force:
        print(f"[SKIP] {args.outfile} already exists. Use --force to overwrite.")
        return 0

    print(f"[INFO] Downloading model from: {args.model_url}")
    print(f"[INFO] Saving to: {args.outfile}")
    download_file(args.model_url, args.outfile)
    print(f"[OK] Saved model → {args.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

