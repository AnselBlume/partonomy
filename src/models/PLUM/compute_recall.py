import os
import json
import glob
from collections import defaultdict
from typing import List, Any, Dict, Tuple
import pandas as pd

def iter_jsonl_files(data_dir: str) -> List[str]:
    """Find all JSONL files under the given directory."""
    return glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)

def to_str_list(gt_list: List[Any]) -> List[str]:
    """Normalize ground-truth list into lowercase strings."""
    out = []
    for x in gt_list:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, (int, float)):
            out.append(str(x))
        elif isinstance(x, dict):
            for v in x.values():
                out.append(str(v))
        elif isinstance(x, list):
            out.extend([str(v) for v in x])
    seen = set()
    canon = []
    for s in out:
        low = s.strip().lower()
        if low and low not in seen:
            seen.add(low)
            canon.append(low)
    return canon

def contains_any(text: str, needles: List[str]) -> bool:
    """Check if any needle is in the text (case-insensitive)."""
    t = (text or "").lower()
    return any(n in t for n in needles)

def eval_sample(sample: Dict[str, Any], max_k: int = 5) -> Tuple[List[List[int]], List[str]]:
    """
    For each <search> call, produce binary flags (gold/not) for top-k retrieved docs.
    Returns:
        flags_per_query: list of binary lists per query
        qtypes: list of qtypes for each query
    """
    calls = sample.get("search_calls", [])
    retrieved = sample.get("retrieved", {})
    items = retrieved.get("item", [])
    if not calls or not items or len(items) != len(calls):
        return [], []
    
    gt_list = to_str_list(sample.get("ground_truth", []))
    flags_per_query = []
    qtypes = []

    for qi, call in enumerate(calls):
        qtype = call.get("qtype", "UNKNOWN")
        qtypes.append(qtype)
        ranked = items[qi] if qi < len(items) else []
        flags = []
        for doc in ranked[:max_k]:
            title = str(doc.get("title", "")).lower()
            content = str(doc.get("content", "")).lower()
            is_gold = contains_any(title, gt_list) or contains_any(content, gt_list)
            flags.append(1 if is_gold else 0)
        flags_per_query.append(flags)

    return flags_per_query, qtypes

def compute_recalls(flags_list: List[List[int]], Ks: List[int]) -> Dict[int, float]:
    """Compute recall@K for each K in Ks."""
    out = {}
    n = len(flags_list)
    if n == 0:
        return {K: 0.0 for K in Ks}
    for K in Ks:
        hits = sum(1 for flags in flags_list if any(flags[:K]))
        out[K] = hits / n
    return out

def main(data_path: str, max_k: int = 5):
    """Main recall computation function."""
    Ks = list(range(1, max_k + 1))
    per_query_flags = []
    per_query_qtypes = []

    # Handle either single file or directory
    # e.g., data_path = "/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/pixar/infoseek/infoseek_retrieve_k_5_interleaved_v1.jsonl"
    if os.path.isfile(data_path):
        files = [data_path]
    else:
        files = iter_jsonl_files(data_path)
    if not files:
        print(f"No JSONL files found in {data_path}")
        return

    total_samples = 0
    total_queries = 0

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_samples += 1
                flags_per_query, qtypes = eval_sample(obj, max_k=max_k)
                for flags, qt in zip(flags_per_query, qtypes):
                    per_query_flags.append(flags)
                    per_query_qtypes.append(qt)
                    total_queries += 1

    # Overall recall
    overall = compute_recalls(per_query_flags, Ks)
    print(f"==== Dataset: {data_path.split('/')[-2]} ====")
    print("\nOverall Recall@K")
    for K in Ks:
        print(f"Recall@{K}: {overall[K]:.4f}")

    # Per-QType recall
    print("\nPer-QType Recall@K")
    qtype_to_flags = defaultdict(list)
    for flags, qt in zip(per_query_flags, per_query_qtypes):
        qtype_to_flags[qt].append(flags)
    for qt, flags_list in sorted(qtype_to_flags.items()):
        recalls = compute_recalls(flags_list, Ks)
        row = " | ".join(f"R@{K}: {recalls[K]:.4f}" for K in Ks)
        print(f"{qt} ({len(flags_list)} queries) -> {row}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute recall@K from retrieved JSONL dataset.")
    parser.add_argument("data_path", type=str, help="Path to JSONL file or directory containing JSONL files.")
    parser.add_argument("--max-k", type=int, default=5, help="Maximum K to compute recall@K for.")
    args = parser.parse_args()

    main(args.data_path, args.max_k)
