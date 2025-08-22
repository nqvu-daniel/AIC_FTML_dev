
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import subprocess, json, os, tempfile
from pathlib import Path

class RunHybridSearch(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="aic_tools/hybrid_search",
            description="Run hybrid + rerank search and mark top results",
        )
    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("query", label="Text query")
        inputs.str("index_dir", label="Artifacts dir", default="./artifacts")
        inputs.int("topk", label="Top-K", default=50, min=1, max=200)
        return types.Property(inputs, view=types.View(label="Hybrid Search"))
    def execute(self, ctx):
        q = ctx.params["query"]
        index_dir = ctx.params.get("index_dir","./artifacts")
        topk = int(ctx.params.get("topk",50))
        # call search script
        cmd = ["python","search_hybrid_rerank.py","--index_dir",index_dir,"--query",q,"--topk",str(topk)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        ctx.log(res.stdout)
        # naive parse to extract (video_id, frame_idx, n)
        lines = [l for l in res.stdout.splitlines() if l.strip() and not l.startswith(("video_id","-"))]
        triples = []
        for l in lines:
            parts = l.split()
            if len(parts) >= 3:
                vid = parts[0]; frame_idx = int(parts[1]); n = int(parts[2])
                triples.append((vid,frame_idx,n))
        # annotate matches
        ds = ctx.dataset
        view = ds.view()
        # create a tag on matched keyframes
        tag = f"hybrid_{q[:16]}".replace(" ","_")
        count = 0
        for vid,frame_idx,n in triples:
            sample = ds.match({"video_id": vid, "frame_idx": frame_idx}).first()
            if sample is not None:
                sample.tags.append(tag); sample.save(); count += 1
        ctx.set_output("count", count)

def register(plugin):
    plugin.register(RunHybridSearch)
