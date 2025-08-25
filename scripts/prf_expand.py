
import argparse, json, re, numpy as np, pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi

def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u1EF9]+", " ", text)
    toks = [t for t in text.split() if len(t) > 1]
    return toks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True, help="text_corpus.jsonl from artifacts")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--fb_docs", type=int, default=10)
    ap.add_argument("--fb_terms", type=int, default=10)
    args = ap.parse_args()

    raw_docs, tokens_list = [], []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            raw_docs.append(j["raw"])
            tokens_list.append(j["tokens"])
    bm25 = BM25Okapi(tokens_list)
    q_tokens = simple_tokenize(args.query)
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(-scores)[:args.fb_docs]

    # RM3-style: pick frequent informative terms from top docs
    from collections import Counter
    c = Counter()
    for i in top_idx:
        c.update(tokens_list[i])
    # remove original query tokens
    for t in q_tokens:
        c.pop(t, None)
    expansions = [w for w,_ in c.most_common(args.fb_terms)]
    expanded = args.query + " " + " ".join(expansions)
    print(expanded)

if __name__ == "__main__":
    main()
