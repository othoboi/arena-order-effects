import os

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "lmarena-ai/arena-human-preference-100k"
KEEP = ["question_id", "model_a", "model_b", "winner", "turn", "anony",  # no convo text, keeps it tiny
        "language", "tstamp", "is_code", "is_refusal", "dedup_tag",
        "judge_hash", "conv_metadata"]


def main():
    files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
    print("parquet files:", files)
    parts = []
    for name in files:
        path = hf_hub_download(REPO, name, repo_type="dataset")
        available = pq.ParquetFile(path).schema_arrow.names
        print(name, "columns:", available)
        cols = [c for c in KEEP if c in available]
        parts.append(pd.read_parquet(path, columns=cols))
    votes = pd.concat(parts, ignore_index=True)

    for col in list(votes.columns):
        first = votes[col].dropna().iloc[0] if votes[col].notna().any() else None
        if isinstance(first, dict):
            flat = pd.json_normalize(votes[col].tolist())
            flat.columns = [f"{col}.{c}" for c in flat.columns]
            def is_plain(v):
                return not isinstance(v, (dict, list, tuple)) and not hasattr(v, "shape")
            simple = [c for c in flat.columns if flat[c].map(is_plain).all()]
            votes = pd.concat([votes.drop(columns=[col]), flat[simple]], axis=1)

    os.makedirs("data", exist_ok=True)
    votes.to_csv("data/votes.csv.gz", index=False, compression="gzip")
    size_mb = os.path.getsize("data/votes.csv.gz") / 1e6
    print(f"saved data/votes.csv.gz: {votes.shape[0]} rows, {votes.shape[1]} columns, {size_mb:.1f} MB")
    print("columns:", list(votes.columns))
    print(votes["winner"].value_counts())


if __name__ == "__main__":
    main()
