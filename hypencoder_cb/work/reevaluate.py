from hypencoder_cb.utils.data_utils import load_qrels_from_ir_datasets
from hypencoder_cb.utils.eval_utils import (
    calculate_metrics_to_file,
    load_standard_format_as_run,
    metric_names_for_dataset,
)
from bm25_rerank import DATASETS, combine_metrics
from typing import List, Optional
from pathlib import Path
import fire


def load_run(retrieval_dir: Path):
    # Older runs saved retrieved_items.jsonl, newer ones run.trec
    if (retrieval_dir / "run.trec").exists():
        run = {}
        with open(retrieval_dir / "run.trec") as f:
            for line in f:
                qid, _, doc_id, _, score, _ = line.split()
                run.setdefault(qid, {})[doc_id] = float(score)
        return run
    return load_standard_format_as_run(retrieval_dir / "retrieved_items.jsonl")


def reevaluate(retrieval_dir: str, ir_dataset_name: str, metric_dir: str) -> None:
    """Re-evaluates one saved retrieval with the dataset's metrics. Leaves
    timing.json in `metric_dir` untouched."""
    print(f"Re-evaluating {retrieval_dir} on {ir_dataset_name}")
    calculate_metrics_to_file(
        load_run(Path(retrieval_dir)),
        load_qrels_from_ir_datasets(ir_dataset_name),
        Path(metric_dir),
        metric_names=metric_names_for_dataset(ir_dataset_name),
    )


def bm25_rerank(
    ret_name: str = "bm25_rerank",
    num_bm25: int = 1000,
    datasets: Optional[List[str]] = None,
) -> None:
    """Re-evaluates every dataset saved by bm25_rerank.py and rebuilds
    results.csv."""
    names = []
    for name, _, ir_dataset_name, _ in DATASETS:
        if datasets is not None and name not in datasets:
            continue
        if not Path(f"retrievals/{ret_name}/{name}").exists():
            print(f"Skipping {name}: no saved retrieval")
            continue
        reevaluate(
            f"retrievals/{ret_name}/{name}",
            ir_dataset_name,
            f"metrics/{ret_name}/{name}",
        )
        names.append(name)

    combine_metrics(ret_name, names, num_bm25)


if __name__ == "__main__":
    fire.Fire({"one": reevaluate, "bm25_rerank": bm25_rerank})
